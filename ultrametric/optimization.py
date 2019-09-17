############################################################################
# Copyright ESIEE Paris (2019)                                             #
#                                                                          #
# Contributor(s) : Giovanni Chierchia, Benjamin Perret                     #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import numpy as np
import higra as hg
import torch as tc
import math
from functools import partial
import scipy.stats as stats 


def subdominant_ultrametric(graph, edge_weights, return_hierarchy=False, dtype=tc.float64):
    """ 
    Subdominant (single linkage) ultrametric of an edge weighted graph.
    
    :param graph: input graph (class ``higra.UndirectedGraph``)
    :param edge_weights: edge weights of the input graph (pytorch tensor, autograd is supported)
    :param return_hierarchy: if ``True``,  the dendrogram representing the hierarchy is also returned as a tuple ``(tree, altitudes)``
    :return: the subdominant ultrametric of the input edge weighted graph (pytorch tensor) (and the hierarchy if ``return_hierarchy`` is ``True``)
    """  
    # compute single linkage if not already provided
    
    tree, altitudes_ = hg.bpt_canonical(graph, edge_weights.detach().numpy())

    # lowest common ancestors of every edge of the graph
    lca_map = hg.attribute_lca_map(tree)
    # the following is used to map lca node indices to their corresponding edge indices in the input graph
    # associated minimum spanning
    mst = hg.get_attribute(tree, "mst")
    # map mst edges to graph edges
    mst_map = hg.get_attribute(mst, "mst_edge_map")
    # bijection between single linkage node and mst edges
    mst_idx = lca_map - tree.num_leaves()
    # mst edge indices in the input graph
    edge_idx = mst_map[mst_idx]

    altitudes = edge_weights[mst_map]
    # sanity check
    # assert(np.all(altitudes.detach().numpy() == altitudes_[tree.num_leaves():]))
    ultrametric = edge_weights[edge_idx]
    
    if return_hierarchy:
        return ultrametric, (tree, tc.cat((tc.zeros(tree.num_leaves(), dtype=dtype), altitudes)))
    else:
        return ultrametric


class UltrametricFitting:
    """
    Fit an ulrametric on an edge weighted graph with respect to a user provided loss function.
    
    The user provided function the loss function must accept 4 arguments: 
    
        - a graph (class ``higra.UnidrectedGraph``);
        - edge weights of the input graph (a pytorch tensor);  
        - an ultrametric on the input graph; and
        - the single linkage clustering of the given ultrametric as a couple ``(tree, altitudes)`` (result of ```higra.bpt_canoncial``).
    
    It must return a pytorch scalar measuring how well the given ultrametric fit the given edge-weighted graph.
    
    """
    
    def __init__(self, epochs, lr, loss, projection='soft', early_stop=True, ultrametric_projection=subdominant_ultrametric):
        """
        :param epochs: Maximum number of epochs run during optimization
        :param lr: learning rate for  the gradient descent
        :param loss: loss function
        :param projection: projection used to impose non negativity of the ultrametric (either 'relu' or 'sofplus').
        :param early_stop: if True the optimization will  end as soon as convergence is assessed (hence perhaps with less epchos than the maximum specified number).
        
        """
        self.lr = lr
        self.epochs = epochs
        self.loss = loss
        self.optimization_callback = []
        self.best = None
        self.best_loss = np.inf
        self.positive = tc.relu if projection == 'relu' else tc.nn.functional.softplus
        self.early_stop = early_stop
        self.ultrametric_projection = ultrametric_projection
        self.max_loss = 0
        
    def add_optimization_callback(self, callback):
        """
        Add a callback that will be called at each iteration of the optimizer.
        The callback must accept 3 parameters: the current UltrametricFitting instance, the iteration number, the current loss.   
        
        :param callback: 
        """
        self.optimization_callback.append(callback)
        
    def fit(self, graph, edge_weights, init=None):
        """
        Fit an ultrametric to the given edge weighted graph
        
        :param graph: input graph
        :param edge_weiths: input graph edge weights
        :param init: optional, initiale value of the estimated ultrametric
        :return: an ultrametric on the input graph
        """
        self._setup(graph, edge_weights, init)
        self._optimization()
        return self._final()

    def _criterion(self):
        M_pos = self.positive(self.M)
        ultrametric, hierarchy = self.ultrametric_projection(self.graph, M_pos, return_hierarchy=True)
        return self.loss(self.graph, self.edge_weights, ultrametric, hierarchy)
        
    def _final(self):
        best_pos = self.positive(self.best)
        return subdominant_ultrametric(self.graph, best_pos).cpu().data.numpy()
        
    def _setup(self, graph, edge_weights, init=None):
        self.graph = graph
        self.edge_weights = tc.from_numpy(edge_weights) 
        # init
        if init is None:
            self.M = subdominant_ultrametric(self.graph, self.edge_weights)
        else:
            self.M = tc.from_numpy(init)
        self.M = self.M.clone().detach().requires_grad_(True)
        #self.optimizer = tc.optim.SGD([self.M], lr=self.lr, momentum=1, nesterov=True) #Adam
        self.optimizer = tc.optim.Adam([self.M], lr=self.lr, amsgrad=True)
        self.history = []
        
    def _optimization(self):
        for t in range(self.epochs):
            self.optimizer.zero_grad()
            loss = self._criterion()
            loss.backward()
            self.optimizer.step()            
            self.history.append(loss.item())      
            if loss < self.best_loss:
                self.best_loss = loss
                self.best = self.M.clone().detach()
            for callback in self.optimization_callback:
                callback(self, t, loss) 
            if self.early_stop and self._has_converged():
                break
                
    def _has_converged(self, window_size1=30,  window_size2=30):
        # okayish convergence assessment by comparing the evolution of the average loss on 2 sliding windows
        h = self.history
        k = len(h)
        if h[k-1] > self.max_loss:
            self.max_loss = h[k-1]
        if k >= window_size1 + window_size2:
            
            m1 = np.mean(h[k - (window_size1 + window_size2):k - window_size2])
            m2 = np.mean(h[k - window_size2:])
            scale = abs(self.max_loss - m2)
            if scale <  0.00001:
                return False
            if abs(m1 - m2) / scale  < 0.005:
                return True
        return False