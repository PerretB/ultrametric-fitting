############################################################################
# Copyright ESIEE Paris (2019)                                             #
#                                                                          #
# Contributor(s) : Giovanni Chierchia, Benjamin Perret                     #
#                                                                          #
# Distributed under the terms of the CECILL-B License.                     #
#                                                                          #
# The full license is in the file LICENSE, distributed with this software. #
############################################################################

import torch
import torch.utils.cpp_extension as cpp
import numpy as np 
import higra
# get location of this file
import os
path = os.path.dirname(os.path.abspath(__file__))



if os.name == 'nt':
    cflags = ['/std:c++14', '/fp:fast', '/GL', '/EHsc', '/MP', '/bigobj', '/O2']
else:
    cflags = ['-D_GLIBCXX_USE_CXX11_ABI=1', '-fabi-version=8', '-ffast-math', '-std=c++14', '-flto', '-O3']

    
# JIT module
cpp_softarea = cpp.load("_softarea", 
                        [os.path.join(path, "softarea.cpp")],
                        extra_cflags=cflags, 
                        verbose=True, 
                        extra_include_paths=[higra.get_include(), 
                                             higra.get_lib_include(), 
                                             np.get_include()])

class SoftareaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, xedge_weights, graph, hierarchy=None, lambda_=1, top_nodes=0):
        edge_weights = xedge_weights.detach().numpy()
       
        if hierarchy is None:
            tree, altitudes = higra.bpt_canonical(graph, edge_weights)
        else:
            tree, altitudes = hierarchy
        
        if type(altitudes) is torch.Tensor:
            altitudes = altitudes.contiguous().detach().numpy()
        
        mst = higra.CptBinaryHierarchy.get_mst(tree)
        # for each edge index ei of the mst, mst_edge_map[ei] gives the index of the edge in the base graph
        mst_edge_map = higra.CptMinimumSpanningTree.get_edge_map(mst)
        op = cpp_softarea.forward(tree, 
                      altitudes, 
                      mst, 
                      mst_edge_map, 
                      lambda_,  
                      top_nodes)
        ctx.intermediate_results = (graph, tree, altitudes, mst, mst_edge_map, lambda_, *op[1:])
        return torch.as_tensor(op[0], dtype=xedge_weights.dtype)

    @staticmethod
    def backward(ctx, grad_output):
        res = cpp_softarea.backward(grad_output.contiguous().detach().numpy(), *ctx.intermediate_results)
        return torch.as_tensor(res, dtype=grad_output.dtype), None, None, None, None

    
class Softarea(torch.nn.Module):
    def __init__(self, graph, lambda_, top_nodes):
        super(Softarea, self).__init__()
        self.reset_parameters()
        self.graph = graph
        self.lambda_ = lambda_
        self.top_nodes = top_nodes

    def reset_parameters(self):
        pass
    
    def forward(self, input):
        return SoftareaFunction.apply(input, self.graph, None, self.lambda_, self.top_nodes)
