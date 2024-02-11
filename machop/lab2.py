import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity

from chop.passes.graph import (
    save_node_meta_param_interface_pass,
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.tools.checkpoint_load import load_model
from chop.ir import MaseGraph

from chop.models import get_model_info, get_model

set_logging_verbosity("info")


########################################################################################################################
"""
Set up the dataset

Here we create a MaseDataModule using the jsc dataset from lab1. Note the MaseDataModule also requires the name of the 
model you plan to use data module with. In this case it is jsc-tiny.
"""
batch_size = 8
model_name = "jsc-tiny"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

########################################################################################################################
"""
Set up the model

Here we use the previously trained jsc-tiny model in lab 1 as an example.
"""
CHECKPOINT_PATH = "/home/lijun/mase/mase_output/jsc-tiny_classification_jsc_2024-02-06/software/training_ckpts/best.ckpt"
model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)

########################################################################################################################
"""
Get a dummy data in

With the dataset module and model information, we can grab an input generator.
"""
# get the input generator
input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

# a demonstration of how to feed an input value to the model
dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

########################################################################################################################
"""
Generate a MaseGraph

We have two forms of passes: transform passes and analysis passes, both of them would require the model to be 
transferred into a MaseGraph to allow manipulation.
"""
# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)

########################################################################################################################
"""
Running an Analysis pass

Analysis pass DOES NOT change the graph

The following analysis passes are essential to prepare the graph for other passes
"""
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)

########################################################################################################################
"""
We will first run a simple graph analysis to understand the structure of the model.
"""
# report graph is an analysis pass that shows you the detailed information in the graph
from chop.passes.graph import report_graph_analysis_pass
_ = report_graph_analysis_pass(mg)

########################################################################################################################
"""
Running another Analysis pass: Profile statistics

The pass profile_statistics_analysis_pass collects statistics of parameters and activations, 
and save them to node's metadata.

Here is a list of all the supported statistics. Refer to the __init__ of statistic classes in 
chop.passes.analysis.statistical_profiler.stat to check the args each stat class takes.

This is a more complex analysis than the previous pass, and thus it would require you to pass in additional arguments 
for this pass.
Example: the range of weights & input activations of nodes

Say we want to collect the tensor-wise min-max range of the 1st torch.nn.Linear nodes' weights & bias, and the 
channel-wise 97% quantile min-max of the 1st torch.nn.Linear nodes' input activations. We can do the following:
"""
pass_args = {
    "by": "type",                                                            # collect statistics by node name
    "target_weight_nodes": ["linear"],                                       # collect weight statistics for linear layers
    "target_activation_nodes": ["relu"],                                     # collect activation statistics for relu layers
    "weight_statistics": {
        "variance_precise": {"device": "cpu", "dims": "all"},                # collect precise variance of the weight
    },
    "activation_statistics": {
        "range_quantile": {"device": "cpu", "dims": "all", "quantile": 0.97} # collect 97% quantile of the activation range
    },
    "input_generator": input_generator,                                      # the input generator for feeding data to the model
    "num_samples": 32,                                                       # feed 32 samples to the model
}

########################################################################################################################
"""
We can use the report_node_meta_param_analysis_pass to inspect the collected statistics.
"""
mg, _ = profile_statistics_analysis_pass(mg, pass_args)
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})

########################################################################################################################
"""
Running a Transform pass: Quantisation

As its name suggests, the transform pass would modify the MaseGraph. Similar to the previous analysis pass example, 
we would need to first declare the configuration for the pass.
"""

pass_args1 = {
    "by": "type",
    "default": {"config": {"name": None}},
    "linear": {
        "config": {
            "name": "integer",
            # data
            "data_in_width": 8,
            "data_in_frac_width": 4,
            # weight
            "weight_width": 8,
            "weight_frac_width": 4,
            # bias
            "bias_width": 8,
            "bias_frac_width": 4,
        }
    },
}

########################################################################################################################
"""
We can then proceed to apply the transformation, in this case, we kept the original graph on purpose, 
so that we can print a diff.
"""
from chop.passes.graph.transforms import (
    quantize_transform_pass,
    summarize_quantization_analysis_pass,
)
from chop.ir.graph.mase_graph import MaseGraph


ori_mg = MaseGraph(model=model)
ori_mg, _ = init_metadata_analysis_pass(ori_mg, None)
ori_mg, _ = add_common_metadata_analysis_pass(ori_mg, {"dummy_in": dummy_in})

mg, _ = quantize_transform_pass(mg, pass_args1)
summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")


########################################################################################################################
# Q4
import pandas as pd
from chop.passes.graph.transforms.quantize.summary import graph_iterator_compare_nodes

df = graph_iterator_compare_nodes(ori_mg, mg, save_path=None, silent=False)
pd.set_option('display.max_columns', None)
print(df)

########################################################################################################################
# Q6
import torch
from chop.passes.graph.utils import get_node_actual_target

for ori_n, quan_n in zip(ori_mg.fx_graph.nodes, mg.fx_graph.nodes):
    ori_target = get_node_actual_target(ori_n)
    quan_target = get_node_actual_target(quan_n)

    if type(ori_target) != type(quan_target):
        print("\nOriginal graph target:", ori_target)
        print("\nQuantised graph target:", quan_target)

        print("\nWeights of original graph:", ori_target.weight)
        print("\nWeights of the quantised graph:", quan_target.weight)

        random_input = torch.randn(quan_target.in_features)
        print("\nRandom input for the graph:\n", random_input)
        print("\nOutput of the original graph:\n", ori_target(random_input))
        print("\nOutput of the quantised graph:\n", quan_target(random_input))

