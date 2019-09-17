############################################################################
# Copyright ESIEE Paris (2019)                                             #
#                                                                          #
# Contributor(s) : Giovanni Chierchia, Benjamin Perret                     #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

from collections import OrderedDict
from sklearn import datasets
import numpy as np
import scipy
from .plots import show_grid, plot_clustering
from .graph import build_graph

def load_datasets(n_samples, n_labeled, preprocess=lambda x: x):
    sets = OrderedDict()
    np.random.seed(2)
    sets['circles'] = create_dataset(n_samples, n_labeled, preprocess, make_circles)
    sets['moons']   = create_dataset(n_samples, 2*n_labeled, preprocess, make_moons)
    sets['blobs']   = create_dataset(n_samples, n_labeled, preprocess, make_blobs)
    sets['varied']  = create_dataset(n_samples, n_labeled, preprocess, make_varied)
    sets['aniso']   = create_dataset(n_samples, n_labeled, preprocess, make_aniso)
    return sets


def show_datasets(sets, show_labeled=False, figname=None):
    get_list = lambda key: [sets[name][key] for name in sets]
    X_list = get_list("X")
    y_list = get_list("y")
    i_list = get_list("labeled") if show_labeled else len(X_list)*[None]
    i_list = [i[0] if i.ndim == 2 else i for i in i_list] #  show only first fold if several exist
    show_grid(plot_clustering, X_list, y_list, i_list, figname=figname)
        
        
#-----------------------------#


def create_dataset(n_samples, n_labeled, preprocess, make_data):
    X, y= make_data(n_samples, n_labeled)
    idx = np.arange(X.shape[0])
    np.random.shuffle(idx)
    data = {
        "X": preprocess(X), 
        "y": y, 
        "n_clusters": len(np.unique(y)), 
        "labeled": idx[:n_labeled],
        "unlabeled": idx[n_labeled:],
    }
    return data

def make_circles(n_samples, n_labeled):
    X, y = datasets.make_circles(n_samples, factor=.5, noise=.05, random_state=10) 
    return X, y

def make_moons(n_samples, n_labeled):
    X, y = datasets.make_moons(n_samples, noise=.05, random_state=42)
    X, y = np.concatenate((X, X + (2.5, 0))), np.concatenate((y, y+2))
    return X, y

def make_blobs(n_samples, n_labeled):
    X, y = datasets.make_blobs(n_samples, random_state=42)
    return X, y

def make_varied(n_samples, n_labeled):
    X, y = datasets.make_blobs(n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=170)
    return X, y

def make_aniso(n_samples, n_labeled):
    X, y = datasets.make_blobs(n_samples, random_state=170)
    X    = np.dot(X, [[0.6, -0.6], [-0.4, 0.8]])
    return X, y