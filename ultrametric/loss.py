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

# To import the function 'loss_dasgupta', a valid C++14 compiler is required.
# On Windows, you should probably run
#   c:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvars64.bat
# to properly setup all environment variables.

def loss_closest(graph, edge_weights, ultrametric, hierarchy):
    """
    Mean square error between ``edge_weights`` and ``ultrametric`` on the input graph :math:`G=(V,E)`:
    
    .. math::
    
        loss = \\frac{1}{|E|}\sum_{e\in E}(edge_weights(e) - ultrametric(e))^2
        
    :param graph: input graph (``higra.UndirectedGraph``)
    :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
    :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
    :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph 
    :return: loss value as a pytorch scalar
    """
    errors = (ultrametric - edge_weights)**2
    return tc.mean(errors)


def loss_cluster_size(graph, edge_weights, ultrametric, hierarchy, top_nodes=0, dtype=tc.float64):
    """
    Cluster size regularization:
    
     .. math::
    
        loss = \\frac{1}{|E|}\sum_{e_{xy}\in E}\\frac{ultrametric(e_{xy})}{\min\{|c|\, | \, c\in Children(lca(x,y))\}}
    
    :param graph: input graph (``higra.UndirectedGraph``)
    :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
    :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
    :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph 
    :param top_nodes: if different from 0, only the top ``top_nodes`` of the hiearchy are taken into account in the cluster size regularization
    :return: loss value as a pytorch scalar
    
    """
    tree, altitudes = hierarchy
    lca_map = hg.attribute_lca_map(tree)
    
    if top_nodes <= 0:
        top_nodes = tree.num_vertices()
    top_nodes = max(tree.num_vertices() - top_nodes, tree.num_leaves())
    top_edges, = np.nonzero(lca_map >= top_nodes)
    
    area = hg.attribute_area(tree)
    min_area = hg.accumulate_parallel(tree, area, hg.Accumulators.min)    
    min_area = min_area[lca_map[top_edges]]
    min_area = tc.tensor(min_area, dtype=dtype)
        
    cluster_size_loss = ultrametric[top_edges] / min_area
    
    return cluster_size_loss.mean()


def loss_closest_and_cluster_size(graph, edge_weights, ultrametric, hierarchy, gamma=1, top_nodes=0, dtype=tc.float64):
    """
    Mean square error between ``edge_weights`` and ``ultrametric`` on the input graph :math:`G=(V,E)` plus cluster size regularization:
    
    :param graph: input graph (``higra.UndirectedGraph``)
    :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
    :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
    :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph 
    :param gamma: weighting of the cluster size regularization (float)
    :param top_nodes: if different from 0, only the top ``top_nodes`` of the hiearchy are taken into account in the cluster size regularization
    :return: loss value as a pytorch scalar
    
    """
    
    closest_loss = loss_closest(graph, edge_weights, ultrametric, hierarchy)
    cluster_size_loss = loss_cluster_size(graph, edge_weights, ultrametric, hierarchy, top_nodes, dtype)
    
    return closest_loss + gamma * cluster_size_loss


def make_pairs(labels, idx):
    """
    Create all triplets from the provided labeld set. This function returns:

    :param labels: provided labels
    :param idx: vertex indices of the provided labels
    :return: a tuple ``pairs=(vertices1, vertices2)`` indexing all the pairs of elements in ``ìdx``
    """
    pairs   = labels[None] == labels[:,None]
    src,dst = np.triu_indices(pairs.shape[0], 1)
    labels  = pairs[src,dst].astype(float)
    pairs   = idx[src], idx[dst]
    return pairs, labels
    
    
def make_triplets(labels, idx):
    """
    Create all triplets from the provided labeld set. This function creates:
    
        - a tuple ``pairs=(vertices1, vertices2)`` describing all the pairs of elements in ``ìdx``
        - a tuple ``(ref, neg)`` decribing couples of pairs of the form ``(i,j)`` and ``(k,j)`` such that ``i`` and ``j``  have the same label and ``k`` and ``j`` have different labels.
    
    :param labels: provided labels
    :param idx: vertex indices of the provided labels
    :return: a tuple ``(pairs, (ref, neg))``
    """
    
    from itertools import combinations
    
    def list_all_triplets():
        #labels = np.array(labels)

        triplets = []
        for label in set(labels):

            # positive indices
            positive = label == labels
            positive_indices, = positive.nonzero()

            if len(positive_indices) < 2:
                continue

            # negative indices
            negative = np.logical_not(positive)
            negative_indices, = negative.nonzero()

            # All positive pairs
            pairs = list(combinations(positive_indices, 2))  

            # Add all negatives for all positive pairs
            for anchor, positive in pairs:
                for negative in negative_indices:
                    triplets += [(anchor, positive, negative)]

        return np.array(triplets)
    
    def get_triplets():
    
        # triplets
        triplets = list_all_triplets()
        anchor   = triplets[:,0]
        positive = triplets[:,1]
        negative = triplets[:,2]

        # linearized indices
        n = labels.size
        idx_table = np.zeros((n, n), dtype=int)
        idx_table[np.triu_indices(n, 1)] = np.arange( n*(n-1)//2 )
        idx_table += idx_table.T
        positive_pairs = idx_table[anchor, positive]
        negative_pairs = idx_table[anchor, negative]

        return positive_pairs, negative_pairs
    
    pairs, _ = make_pairs(labels, idx)
    pos, neg = get_triplets()
    
    return pairs, (pos, neg)
    
    
def loss_triplet(graph, edge_weights, ultrametric, hierarchy, triplets, margin):
    """
    Triplet loss regularization with triplet :math:`\mathcal{T}`:
    
     .. math::
    
        loss = \sum_{(ref, pos, neg)\in \mathcal{T}} \max(0, ultrametric(ref, pos) - ultrametric(ref, neg) + margin)
    
    :param graph: input graph (``higra.UndirectedGraph``)
    :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
    :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
    :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph 
    :param triplets:
    :param margin:
    :return: loss value as a pytorch scalar
    """
    tree, altitudes = hierarchy
    #mst = hg.get_attribute(tree, "mst")
    #mst_map = hg.get_attribute(mst, "mst_edge_map")
    lcaf = hg.make_lca_fast(tree)
    
    #closest_loss = (ultrametric - edge_weights)**2
    pairs, (pos, neg) = triplets
    pairs_distances = altitudes[lcaf.lca(*pairs)] # ultrametric[mst_map[lcaf.lca(*pairs) - tree.num_leaves()]]
    
    triplet_loss = tc.relu(pairs_distances[pos] - pairs_distances[neg] + margin)
    
    return triplet_loss.mean()

    
def loss_closest_and_triplet(graph, edge_weights, ultrametric, hierarchy, triplets, margin, gamma=1):
    """
    Mean square error between ``edge_weights`` and ``ultrametric`` on the input graph :math:`G=(V,E)` plus triplet loss regularization with triplet :math:`\mathcal{T}`:
    
     .. math::
    
        loss = \\frac{1}{|E|}\sum_{e_{xy}\in E}(edge_weights(e_{xy}) - ultrametric(e_{xy}))^2 + \gamma \sum_{(ref, pos, neg)\in \mathcal{T}} \max(0, ultrametric(ref, pos) - ultrametric(ref, neg) + margin)
    
    :param graph: input graph (``higra.UndirectedGraph``)
    :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
    :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
    :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph 
    :param triplets:
    :param margin:
    :param gamma: weighting of the cluster size regularization (float)
    :return: loss value as a pytorch scalar
    """
    closest_loss = loss_closest(graph, edge_weights, ultrametric, hierarchy)
    triplet_loss = loss_triplet(graph, edge_weights, ultrametric, hierarchy, triplets, margin)
    
    return closest_loss + gamma * triplet_loss


def loss_dasgupta(graph, edge_weights, ultrametric, hierarchy, sigmoid_param=5, mode='dissimilarity'):
    """
    Relaxation of cost function defined in S. Dasgupta, A cost function for similarity-based hierarchical clustering, 2016.
    
    :param graph: input graph (``higra.UndirectedGraph``)
    :param edge_weights: edge weights of the input graph (``torch.Tensor``, autograd is supported)
    :param ultrametric; ultrametric on the input graph  (``torch.Tensor``, autograd is supported)
    :param hierarchy: optional,  if provided must be a tuple ``(tree, altitudes)`` corresponding to the result of ``higra.bpt_canonical`` on the input edge weighted graph 
    :param sigmoid_param: scale parameter used in the relaxation of the cluster size relaxation
    :param gamma: weighting of the cluster size regularization (float)
    :return: loss value as a pytorch scalar
    """
    
    # The following line requires that a valid C++14 compiler be installed.
    # On Windows, you should probably run
    #   c:\Program Files (x86)\Microsoft Visual Studio\2017\Enterprise\VC\Auxiliary\Build\vcvars64.bat
    # to properly setup all environment variables
    from .softarea import SoftareaFunction
    
    # hierarchy: nodes are sorted by altitudes (from leaves to the root)
    tree, altitudes = hierarchy
    
    # softarea
    area = SoftareaFunction.apply(ultrametric, graph, hierarchy, sigmoid_param)
    
    # lowest common ancestor
    lca = hg.attribute_lca_map(tree)
        
    # cost function
    if mode == 'similarity':
        loss = area[lca] * edge_weights
    elif mode == 'dissimilarity':
        loss = area[lca] / edge_weights
    else:
        raise Exception("'mode' can only be 'similarity' or 'dissilarity'")
    
    return loss.mean()