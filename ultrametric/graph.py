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
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.csgraph import minimum_spanning_tree
from .plots import show_grid, plot_graph
   

def build_graph(X, graph_type='knn-mst', n_neighbors=5, mst_weight=1):
     
    if graph_type == 'complete':
        d = pdist(X)
        A = squareform(d)
        
    elif graph_type == 'knn':
        A = kneighbors_graph(X, n_neighbors, mode='distance').toarray()
        A = (A + A.T) / 2
        
    elif graph_type == 'knn-mst':
        A = kneighbors_graph(X, n_neighbors, mode='distance').toarray()
        A = (A + A.T) / 2
        D = squareform(pdist(X))
        MST = minimum_spanning_tree(D).toarray()
        MST = (MST + MST.T) / 2
        A = np.maximum(A, MST * mst_weight)
        
    elif graph_type == 'mst':
        D = squareform(pdist(X))
        A = minimum_spanning_tree(D).toarray()
        A = A + A.T  
        
    return A


def show_graphs(sets, graph_type='knn-mst', figname=None, **args):
    get_list = lambda key: [sets[name][key] for name in sets]
    X_list = get_list("X")
    y_list = get_list("y")
    A_list = [build_graph(X, graph_type, **args) for X in X_list]
    show_grid(plot_graph, A_list, X_list, y_list, figname=figname)
