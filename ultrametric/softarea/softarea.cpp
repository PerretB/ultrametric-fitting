/***************************************************************************
* Copyright ESIEE Paris (2019)                                             *
*                                                                          *
* Contributor(s) : Giovanni Chierchia, Benjamin Perret                     *
*                                                                          *
* Distributed under the terms of the CECILL-B License.                     *
*                                                                          *
* The full license is in the file LICENSE, distributed with this software. *
****************************************************************************/

#include <higra/graph.hpp>
#include <higra/attribute/tree_attribute.hpp>
#include <higra/accumulator/tree_accumulator.hpp>
#include <vector>
#include "xtl/xmeta_utils.hpp"
#include "pybind11/pybind11.h"
#include <pybind11/stl_bind.h>
#include "pybind11/functional.h"
#include "xtensor/xeval.hpp"
#include "xtensor/xindex_view.hpp"
#include <iostream>
#define FORCE_IMPORT_ARRAY
#include "xtensor-python/pyarray.hpp"
#include "xtensor-python/pytensor.hpp"

PYBIND11_MAKE_OPAQUE(std::vector<hg::index_t>);

template<typename T>
using pyarray = xt::pyarray<T>;
template<typename T>
using pyarray = xt::pyarray<T>;

namespace py = pybind11;

double sigmoid(double x, double lambda){
    return 1 / (1 + exp(-lambda * x));
}

template <typename T>
auto sigmoid(const xt::xexpression<T>& xx, double lambda){
    auto & x = xx.derived_cast();
    return xt::eval(1 / (1 + xt::exp(-lambda * x)));
}


auto dsigmoid(double x, double lambda){
    auto s = sigmoid(x, lambda);
    return lambda * s * (1 - s);
}

template <typename T>
auto dsigmoid(const xt::xexpression<T>& x, double lambda){
    auto s = sigmoid(x, lambda);
    return xt::eval(lambda * s * (1 - s));
}



using namespace hg;

auto softarea_forward(const tree & tree, 
                      const xt::pytensor<double, 1> &  altitudes, 
                      const ugraph& mst, 
                      const xt::pytensor<hg::index_t, 1>& mst_edge_map, 
                      const double lambda=1,  
                      index_t top_nodes=0) {
    pyarray<double> res =  xt::eval(xt::zeros_like(altitudes));
    auto & parent = parents(tree);
        
    auto area = attribute_area(tree);
    auto siblings = attribute_sibling(tree);
    auto altitudes_parent = propagate_parallel(tree, altitudes);
    index_t num_v = num_vertices(tree);
    index_t nb = num_leaves(tree);
    array_1d<double> soft_area = xt::zeros<double>({(size_t)num_v});
    auto rootn = root(tree);
    auto transfer_function_at_zero = sigmoid(0, lambda);
    top_nodes = std::min(num_v, top_nodes);
    std::vector<index_t> requested_nodes{};
    

    if (top_nodes > 0 && top_nodes != num_v){
            
        auto node_limit = num_v - top_nodes;
            
        requested_nodes.push_back(rootn);
        for(index_t n=node_limit; n < num_v; n++){
            for(auto c: children_iterator(n, tree)){
                requested_nodes.push_back(c);
            }
        }
        
        for(auto r: requested_nodes){
            auto a = altitudes(r);
            index_t ref_leaves[2];
            int num_refs = 0;
            double sa;
            if(is_leaf(r, tree)){
                sa = transfer_function_at_zero;
                ref_leaves[0] = r;
                num_refs = 1;
            } else {
                sa = 2 * sigmoid(a, lambda);
                auto e = edge_from_index(r - nb, mst);
                ref_leaves[0] = source(e, mst);
                ref_leaves[1] = target(e, mst);
                num_refs = 2;
            }
            
            for(int i = 0; i < num_refs; i++){
                for(index_t n = ref_leaves[i]; n != rootn; n = parent(n)){
                    sa += sigmoid(a - altitudes_parent(n), lambda) * area(siblings(n));
                }
            }
            
            soft_area(r) = sa;
            if(num_refs == 2){
                soft_area(r) *= 0.5;
            }
        }
    }else{
        xt::view(soft_area, xt::range(0, nb)) = transfer_function_at_zero;
        auto trans_altitudes_parent = sigmoid(-altitudes_parent, lambda);
        auto sib_weights = trans_altitudes_parent * xt::index_view(area, siblings);
            
        array_1d<double> tmp = xt::zeros<double>({num_v});
        for(auto i: hg::root_to_leaves_iterator(tree, hg::leaves_it::include, hg::root_it::exclude)){
            auto par = hg::parent(i, tree);
            tmp(i) += tmp(par);
            tmp(i) += sib_weights(i);
        }
            
        xt::view(soft_area, xt::range(0, nb)) += xt::view(tmp, xt::range(0, nb));
            
        for(auto r: root_to_leaves_iterator(tree, leaves_it::exclude)){
            auto a = altitudes(r);
            auto sa = 2 * sigmoid(a, lambda);
            auto e = edge_from_index(r - nb, mst);
            index_t ref_leaves[2];
            ref_leaves[0] = source(e, mst);
            ref_leaves[1] = target(e, mst);

            for(int i = 0; i < 2; i++){
                for(index_t n = ref_leaves[i]; n != rootn; n = parent(n)){
                    sa += sigmoid(a - altitudes_parent(n), lambda) * area(siblings(n));
                }
            }
            soft_area(r) = sa * 0.5;
        }
    }
            
    return py::make_tuple(std::move(soft_area), std::move(siblings), std::move(altitudes_parent), std::move(area), std::move(requested_nodes));
}

auto softarea_backward(
    const xt::pytensor<double, 1> & grad_output,
    const ugraph & graph,
    const tree & tree, 
    const xt::pytensor<double, 1> &  altitudes, 
    const ugraph& mst,
    const xt::pytensor<hg::index_t, 1>& mst_edge_map,
    const double lambda,  
    const xt::pytensor<hg::index_t, 1>& siblings, 
    const xt::pytensor<double, 1>& altitudes_parent,
    const xt::pytensor<index_t, 1>& area,
    const std::vector<index_t>& requested_nodes) {
    
        
    xt::pytensor<double, 1> grad_input = xt::zeros<double>({num_edges(graph)});
    auto & parent = parents(tree);    
    index_t nb = num_leaves(tree);
    auto rootn = root(tree);
    xt::pytensor<double, 1> go2(grad_output);
    xt::view(go2, xt::range(nb, num_vertices(tree))) *= 0.5;
        
    if(requested_nodes.size() > 0){
        for(auto r: requested_nodes){
            index_t ref_leaves[2];
            int num_refs = 0;
            auto a = altitudes(r);
            if(is_leaf(r, tree)){
                ref_leaves[0] = r;
                num_refs = 1;
            } else {
                
                auto e = edge_from_index(r - nb, mst);
                ref_leaves[0] = source(e, mst);
                ref_leaves[1] = target(e, mst);
                num_refs = 2;
                
                grad_input(mst_edge_map(r - nb)) += 2 * dsigmoid(a, lambda) * go2(r);
            }
            
            for(int i = 0; i < num_refs; i++){
                double total = 0;
                for(index_t n = ref_leaves[i]; n != rootn; n = parent(n)){
                    double tmp = dsigmoid(a - altitudes_parent(n), lambda) * area(siblings(n));
                    total += tmp;
                    grad_input(mst_edge_map(parent(n)- nb)) -= tmp * go2(r);
                }
                if(!is_leaf(r, tree)){
                    grad_input(mst_edge_map(r - nb)) += total * go2(r);
                }
            }  
        }
    } else {
        auto trans_altitudes_parent = dsigmoid(-altitudes_parent, lambda);
        auto tmp = xt::eval(trans_altitudes_parent * xt::index_view(area, siblings));
        for(auto r: leaves_iterator(tree)){
            for(index_t n = r; n != rootn; n = parent(n)){
                grad_input(mst_edge_map(parent(n) - nb)) -= tmp(n) * go2(r);
            }
        }
        for(auto r: root_to_leaves_iterator(tree, leaves_it::exclude)){
            auto a = altitudes(r);
            auto e = edge_from_index(r - nb, mst);
            index_t ref_leaves[2];
            ref_leaves[0] = source(e, mst);
            ref_leaves[1] = target(e, mst);
            
            grad_input(mst_edge_map(r - nb)) += 2 * dsigmoid(a, lambda) * go2(r);
                
            for(int i = 0; i < 2; i++){
                double total = 0;
                for(index_t n = ref_leaves[i]; n != rootn; n = parent(n)){
                    double tmp = dsigmoid(a - altitudes_parent(n), lambda) * area(siblings(n));
                    total += tmp;
                    grad_input(mst_edge_map(parent(n)- nb)) -= tmp * go2(r);
                }
                
                grad_input(mst_edge_map(r - nb)) += total * go2(r);
                
            } 
        }
    }  
    return grad_input;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    xt::import_numpy();
    py::module::import("higra");
    py::bind_vector<std::vector<hg::index_t>>(m, "VectorIndex");
    m.def("forward", &softarea_forward, "softarea forward");
    m.def("backward", &softarea_backward, "softarea backward");
}