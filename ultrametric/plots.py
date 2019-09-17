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
import matplotlib.colors as mc
from matplotlib.collections import LineCollection
import colorsys
from scipy.cluster.hierarchy import dendrogram, set_link_color_palette


COLORS  = np.array(['#377eb8', '#ff7f00', '#4daf4a', '#a65628', '#f781bf', '#984ea3', '#999999', '#e41a1c', '#dede00'])
MARKERS = np.array(['o', '^', 's', 'X'])


def show_grid(plot_fun, *args, figname=None):
    n = len(args[0])
    fig = plt.figure(figsize=(n * 2 + 3, 2.5))
    plt.subplots_adjust(left=.02, right=.98, bottom=.001, top=.96, wspace=.05, hspace=.01)
    for i, items in enumerate(zip(*args)):
        plt.subplot(1, n, i+1)
        plot_fun(*items)
    plt.show()
    if figname:
        fig.savefig(figname, dpi=500, pad_inches=0, bbox_inches='tight')
    
        
def plot_clustering(X, y, idx=None):
    ec = COLORS[y%len(COLORS)]
    plt.scatter(X[:, 0], X[:, 1], s=15, linewidths=1.5, c=lighten_color(ec), edgecolors=ec, alpha=0.9)
    #plt.axis([X[:,0].min(), X[:,0].max(), X[:,1].min(), X[:,1].max()])
    plt.xticks(())
    plt.yticks(())
    if idx is not None:
        iec = COLORS[y[idx]%len(COLORS)]
        plt.scatter(X[idx,0], X[idx,1], s=30, color=iec, marker='s', edgecolors='k')
    
    
def plot_graph(adjacency, X, y): 
    non_zero = (np.triu(adjacency, k=1) > 0)
    sources, targets = np.where(non_zero)
    segments = np.stack((X[sources, :], X[targets, :]), axis=1)
    lc = LineCollection(segments, zorder=0, colors='k')
    lc.set_linewidths(1)
    ax = plt.gca()
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_xlim(segments[:,:,0].min(), segments[:,:,0].max())
    ax.set_ylim(segments[:,:,1].min(), segments[:,:,1].max())
    ax.add_collection(lc)
    #plt.axis('equal')
    ax.scatter(X[:, 0], X[:, 1], s=20, c='w', edgecolors='k')
    
    
def plot_dendrogram(linkage_matrix, n_clusters=0, lastp=30):
    extra = {} if lastp is None else dict(truncate_mode='lastp', p=lastp)
    set_link_color_palette(list(COLORS))
    dsort = np.sort(linkage_matrix[:,2]) 
    dendrogram(linkage_matrix, no_labels=True, above_threshold_color="k", color_threshold = dsort[-n_clusters+1], **extra)
    plt.yticks([])
    
    
def lighten_color(color_list, amount=0.25):
    """
    Lightens the given color by multiplying (1-luminosity) by the given amount.
    Input can be matplotlib color string, hex string, or RGB tuple.

    Examples:
    >> lighten_color('g', 0.3)
    >> lighten_color('#F034A3', 0.6)
    >> lighten_color((.3,.55,.1), 0.5)
    """    
    out = []
    for color in color_list:
        try:
            c = mc.cnames[color]
        except:
            c = color
        c = colorsys.rgb_to_hls(*mc.to_rgb(c))
        lc = colorsys.hls_to_rgb(c[0], 1 - amount * (1 - c[1]), c[2])
        out.append(lc)
    return out


def plot_score_bars(scores, deviation=None, experiment_labels=None, set_labels=None, figname=None):
    SMALL_SIZE = 14
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 18
    
    fig = plt.figure(figsize=(12,6))
    plt.gca().yaxis.grid(True)
    plt.gca().set_axisbelow(True)
    
    nmethods, nsets = scores.shape
    barWidth = 0.8 / nmethods
    
    for i in range(nmethods):
        pos = np.arange(nsets) + barWidth * i
        m_name = None if experiment_labels is None else experiment_labels[i]
        dev = None if deviation is None else deviation[i, :]
        rects = plt.bar(pos, scores[i, :], color=COLORS[i%len(COLORS)], width=barWidth, edgecolor='w', label=m_name, yerr=dev)
        for rect in rects:
            height = rect.get_height()
            plt.text(rect.get_x() + rect.get_width()/2. - 0.02, height-0.02, '%2.2f' % (height), ha='center', va='top', rotation=-90, color='w', fontsize=MEDIUM_SIZE, fontweight='bold')
     
    # Add xticks on the middle of the group bars
    if set_labels is not None:
        plt.xlabel('Dataset', fontweight='bold')
        plt.xticks([r + barWidth for r in range(nsets)], set_labels)

    # Trim the bottom and extend the top a little bit
    plt.ylim(0.2, 1.07)
    
    # Adjust font
    plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
    plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
    plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

    # Create legend & Show graphic
    plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.05), ncol=3, fancybox=True, shadow=True)
    plt.show()
    
    # Save on disk
    if figname:
        fig.savefig(figname, dpi=500, pad_inches=0, bbox_inches='tight')
    

def plot_clustering_v2(X, y, idx=None):
    ec  = COLORS[y%len(COLORS)]
    iec = COLORS[y[idx]%len(COLORS)]
    for i, mi in zip(np.unique(y), MARKERS):
        mask = i == y
        plt.scatter(X[mask,0], X[mask,1], s=40, c=lighten_color(ec[mask]), edgecolors=ec[mask], alpha=0.8, linewidths=2, marker=mi)
        if len(idx)>0:
            plt.scatter(X[idx[i==y[idx]],0], X[idx[i==y[idx]],1], s=40, color=iec[i==y[idx]], edgecolors='k', marker=mi)
    plt.xticks(())
    plt.yticks(())
    
def plot_loss(loss):
    plt.plot(loss)
    plt.yticks(())