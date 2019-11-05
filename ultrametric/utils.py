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
from .plots import show_grid, plot_clustering, plot_dendrogram, plot_loss, plot_score_bars
import numpy as np
        
class Experiments:
    
    def __init__(self, sets):
        self.sets = sets
        self.results = OrderedDict()
        self.shower = {}
        
        get_list = lambda key, sets: [sets[set_name][key] for set_name in sets]
        
        self.shower["clustering"] = lambda name: (plot_clustering, get_list("X", self.sets), get_list("y", self.results[name]))
        self.shower["dendrogram"] = lambda name: (plot_dendrogram, get_list("linkage", self.results[name]), get_list("n_clusters", self.sets))
        self.shower["loss"] = lambda name: (plot_loss, get_list("loss", self.results[name]))
        self.shower["embedding"] = lambda name: (plot_clustering, get_list("embedding", self.results[name]), get_list("y", self.results[name]))
        
    def get_data(self, set_name, *others):
        names = ["X", "y", "n_clusters", "labeled"] + list(others)
        o = [self.sets[set_name].get(n, None) for n in names]
        return o
    
    def add_results(self, experiment_name, set_name, **kwargs):
        assert set_name in self.sets.keys()
        e = self.results.setdefault(experiment_name, OrderedDict())
        r = e.setdefault(set_name, OrderedDict())
        e[set_name].update(kwargs)
        
    def show(self, name, what=["clustering"]):
        for w in what:
            show_grid(*self.shower[w](name))
            
    def compare(self, score_index, experiment_names=None, set_names=None):
        if set_names  is None:
            set_names = list(self.sets.keys())
        if experiment_names is None:
            experiment_names = list(self.results.keys())
        scores = np.zeros((len(experiment_names), len(set_names)))
        scores_err = np.zeros_like(scores)
        
        for i, m in enumerate(experiment_names):
            for j, s in enumerate(set_names):
                scores[i,j] = self.results[m][s]["scores"][score_index]
                scores_err[i,j] = self.results[m][s]["scores_dev"][score_index]
                    
        plot_score_bars(scores, deviation=scores_err, experiment_labels=experiment_names, set_labels=set_names)
                          
            
