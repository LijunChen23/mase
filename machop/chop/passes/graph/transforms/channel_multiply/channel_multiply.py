from copy import copy, deepcopy
import logging
import torch
from chop.passes.graph.interface.save_and_load import load_mase_graph_interface_pass

from ...utils import (
    deepcopy_mase_graph,
    get_mase_op,
    get_mase_type,
    get_node_actual_target,
    get_parent_name,
    get_similar_node_actual_target,
    match_a_pattern,
    get_node_target_by_name,
)

from ..quantize.modify import create_new_fn, create_new_module
from ..quantize.quant_parsers import parse_node_config, relink_node_meta, update_quant_meta_param

logger = logging.getLogger(__name__)

QUANTIZEABLE_OP = (
    "add",
    "bmm",
    "conv1d",
    "conv2d",
    "matmul",
    "mul",
    "linear",
    "relu",
    "sub",
)


def get_config(config: dict, name: str):
    if name in config:
        return config[name]["config"]
    else:
        return config["default"]["config"]


# def update_quant_meta_param(*args, **kwargs):
#     # TODO: remove this function when the add_common_metadata is fixed
#     pass


def graph_iterator_quantize_by_type(graph, config: dict):
    # Some modules might need information from two graphs to be initilized
    if (
        config.get("baseline_weight_path") is not None
        and config.get("load_type") == "mz"
    ):
        bl_graph = deepcopy_mase_graph(graph)
        bl_graph = load_mase_graph_interface_pass(
            bl_graph, pass_args=config.get("baseline_weight_path")
        )
    else:
        bl_graph = None
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, get_mase_op(node))
        if node_config["name"] is None:
            continue
        node_config = parse_node_config(node_config, get_mase_op(node))
        # if get_mase_type(node) == "module":
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            successor_module = get_similar_node_actual_target(
                bl_graph, node.next
            )  # Certain modules will require information about their successor module to complete the initialization process. (For LogicNets, activation functions are needed.)
            bl_module = get_similar_node_actual_target(bl_graph, node)
            new_module = create_new_module(
                get_mase_op(node),
                ori_module,
                node_config,
                node.meta,
                bl_module,
                successor_module,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            # update precision and type in meta.parameters["common"]
            update_quant_meta_param(node, node_config, get_mase_op(node))
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = copy(node.meta["mase"])
                # new_node.meta["mase"].node -> new_node
                relink_node_meta(new_node, model=graph.model)
                update_quant_meta_param(new_node, node_config, get_mase_op(node))
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
    return graph


def graph_iterator_quantize_by_name(graph, config: dict):
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        node_config = get_config(config, node.name)
        if node_config["name"] is None:
            continue
        node_config = parse_node_config(node_config, get_mase_op(node))
        output_layers_names = node_config.get("additional_layers_outputs", [])
        output_layers = [
            get_node_target_by_name(graph, name) for name in output_layers_names
        ]
        input_layers_names = node_config.get("additional_layers_inputs", [])
        input_layers = [
            get_node_target_by_name(graph, name) for name in input_layers_names
        ]
        if node.op == "call_module":
            ori_module = get_node_actual_target(node)
            new_module = create_new_module(
                get_mase_op(node),
                ori_module,
                node_config,
                node.meta,
                input_layers=input_layers,
                output_layers=output_layers,
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            update_quant_meta_param(node, node_config, get_mase_op(node))
            logger.debug(f"Quantized module: {node.target} with config: {node_config}")
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = copy(node.meta["mase"])
                relink_node_meta(new_node, model=graph.model)
                update_quant_meta_param(new_node, node_config, get_mase_op(node))
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
            logger.debug(
                f"Quantized function: {node.target} with config: {node_config}"
            )
        else:
            raise ValueError(
                "Unsupported node type for quantisation: {}".format(get_mase_type(node))
            )
    return graph


def graph_iterator_quantize_by_regex_name(graph, config: dict):
    patterns = list(config.keys())
    for node in graph.fx_graph.nodes:
        if get_mase_op(node) not in QUANTIZEABLE_OP:
            continue
        matched_pattern = match_a_pattern(node.name, patterns)
        if not matched_pattern:
            node_config = get_config(config, "default")
        else:
            node_config = get_config(config, matched_pattern)
        if node_config["name"] is None:
            continue
        node_config = parse_node_config(node_config, get_mase_op(node))
        # if get_mase_type(node) == "module":
        if node.op == "call_module":
            ori_module = graph.modules[node.target]
            new_module = create_new_module(
                get_mase_op(node), ori_module, node_config, node.meta
            )
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
            update_quant_meta_param(node, node_config, get_mase_op(node))
        elif get_mase_type(node) in [
            "builtin_func",
            "module_related_func",
        ]:
            new_f, args, kwargs = create_new_fn(node, node_config)
            with graph.fx_graph.inserting_before(node):
                new_node = graph.fx_graph.call_function(new_f, args, kwargs)
                new_node.name = node.name
                new_node.meta["mase"] = deepcopy(node.meta["mase"])
                relink_node_meta(new_node, model=graph.model)
                update_quant_meta_param(new_node, node_config, get_mase_op(node))
                node.replace_all_uses_with(new_node)
            graph.fx_graph.erase_node(node)
        else:
            raise ValueError(
                "Unsupported node type for quantisation:{}".format(get_mase_type(node))
            )
    return graph


def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return torch.nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)

def instantiate_ReLU(inplace):
    return torch.nn.ReLU(inplace=inplace)

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:  # node.name = e.g. seq_blocks_2
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']  # e.g., {'name': 'output_only', 'channel_multiplier': 2}
        name = config.get("name", None)  # e.g., "both", "input_only", "output_only"

        if name is not None:
            ori_module = graph.modules[node.target]  # e.g., node.target = "seq_blocks.4"
            if isinstance(ori_module, torch.nn.ReLU):
                inplace = ori_module.inplace
                if name == "inplace":
                    inplace = inplace * config["channel_multiplier"]
                new_module = instantiate_ReLU(inplace)
            elif isinstance(ori_module, torch.nn.Linear):
                in_features = ori_module.in_features     # e.g., in_features = 16
                out_features = ori_module.out_features   # e.g., out_features = 5
                bias = ori_module.bias
                if name == "output_only":
                    out_features = out_features * config["channel_multiplier"]
                elif name == "both":
                    in_features = in_features * config["channel_multiplier_input"]
                    out_features = out_features * config["channel_multiplier_output"]
                elif name == "input_only":
                    in_features = in_features * config["channel_multiplier"]
                new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)  # parent_name = seq_blocks, name = e.g. 3
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}
