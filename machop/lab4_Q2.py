import sys
import logging
import os
from pathlib import Path
from pprint import pprint as pp

# figure out the correct path
#machop_path = Path(".").resolve().parent.parent /"machop"
machop_path = Path("/home/lijun/mase/machop")
assert machop_path.exists(), "Failed to find machop at: {}".format(machop_path)
sys.path.append(str(machop_path))

from chop.dataset import MaseDataModule, get_dataset_info
from chop.tools.logger import set_logging_verbosity, get_logger

from chop.passes.graph.analysis import (
    report_node_meta_param_analysis_pass,
    profile_statistics_analysis_pass,
)
from chop.passes.graph import (
    add_common_metadata_analysis_pass,
    init_metadata_analysis_pass,
    add_software_metadata_analysis_pass,
)
from chop.tools.get_input import InputGenerator
from chop.ir.graph.mase_graph import MaseGraph

from chop.models import get_model_info, get_model

set_logging_verbosity("info")

logger = get_logger("chop")
logger.setLevel(logging.INFO)

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

model_info = get_model_info(model_name)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

dummy_in = {"x": next(iter(data_module.train_dataloader()))[0]}


########################################################################################################################
"""
This time we are going to use a slightly different network, so we define it as a Pytorch model.
"""
from torch import nn
from chop.passes.graph.utils import get_parent_name

# define a new model
"""
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear  2
            nn.Linear(16, 16),  # linear  3
            nn.Linear(16, 5),   # linear  4
            nn.ReLU(5),  # 5
        )

    def forward(self, x):
        return self.seq_blocks(x)
"""

# define a new model
class JSC_Three_Linear_Layers(nn.Module):
    def __init__(self):
        super(JSC_Three_Linear_Layers, self).__init__()
        self.seq_blocks = nn.Sequential(
            nn.BatchNorm1d(16),  # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 16),  # linear seq_2
            nn.ReLU(16),  # 3
            nn.Linear(16, 16),  # linear seq_4
            nn.ReLU(16),  # 5
            nn.Linear(16, 5),  # linear seq_6
            nn.ReLU(5),  # 7
        )

    def forward(self, x):
        return self.seq_blocks(x)




model = JSC_Three_Linear_Layers()

# generate the mase graph and initialize node metadata
mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)


########################################################################################################################
"""
Model Architecture Modification as a Transformation Pass

Similar to what you have done in lab2, one can also implement a change in model architecture as a transformation pass:
"""
def instantiate_linear(in_features, out_features, bias):
    if bias is not None:
        bias = True
    return nn.Linear(
        in_features=in_features,
        out_features=out_features,
        bias=bias)


def instantiate_ReLU(inplace):
    return nn.ReLU(inplace=inplace)


def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    print("main_config:\n",main_config)
    default = main_config.pop('default', None)
    print("default:\n",default)
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
            print("ori_module:\n", ori_module)
            if isinstance(ori_module, nn.ReLU):
                inplace = ori_module.inplace
                if name == "inplace":
                    inplace = inplace * config["channel_multiplier"]
                print("inplace:\n", inplace)
                new_module = instantiate_ReLU(inplace)
                print("new_module:\n", new_module)
            elif isinstance(ori_module, nn.Linear):
                in_features = ori_module.in_features     # e.g., in_features = 16
                out_features = ori_module.out_features   # e.g., in_features = 5
                bias = ori_module.bias
                if name == "output_only":
                    out_features = out_features * config["channel_multiplier"]
                elif name == "both":
                    in_features = in_features * config["channel_multiplier"]
                    out_features = out_features * config["channel_multiplier"]
                elif name == "input_only":
                    in_features = in_features * config["channel_multiplier"]
                new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)  # parent_name = seq_blocks, name = e.g. 3
            setattr(graph.modules[parent_name], name, new_module)
        print("\n")
    return graph, {}


"""
pass_config = {
    "by": "name",
    "default": {"config": {"name": None}},
    "seq_blocks_2": {
        "config": {
            "name": "output_only",
            # weight
            "channel_multiplier": 2,
        }
    },
    "seq_blocks_3": {
        "config": {
            "name": "both",
            "channel_multiplier": 2,
        }
    },
    "seq_blocks_4": {
        "config": {
            "name": "input_only",
            "channel_multiplier": 2,
        }
    },
}
"""

pass_config = {
    "by": "name",
    "default": {"config": {"name": None}},
    "seq_blocks_2": {
        "config": {
            "name": "output_only",
            # weight
            "channel_multiplier": 2,
        }
    },
    "seq_blocks_3": {
        "config": {
            "name": "inplace",
            "channel_multiplier": 2,
        }
    },
    "seq_blocks_4": {
        "config": {
            "name": "both",
            "channel_multiplier": 2,
        }
    },
    "seq_blocks_5": {
        "config": {
            "name": "inplace",
            "channel_multiplier": 2,
        }
    },
    "seq_blocks_6": {
        "config": {
            "name": "input_only",
            "channel_multiplier": 2,
        }
    },
}


"""# this performs the architecture transformation based on the config
mg, _ = redefine_linear_transform_pass(
    graph=mg, pass_args={"config": pass_config})"""


"""
#mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
#mg, _ = add_software_metadata_analysis_pass(mg, None)

from chop.passes.graph import report_graph_analysis_pass
_ = report_graph_analysis_pass(mg)"""




import copy


search_spaces = []
channel_multipliers = [2, 4, 8]
base_pass_config = {
    "by": "name",
    "default": {"config": {"name": None}},
    "seq_blocks_2": {"config": {"name": "output_only"}},
    "seq_blocks_3": {"config": {"name": "inplace"}},
    "seq_blocks_4": {"config": {"name": "both"}},
    "seq_blocks_5": {"config": {"name": "inplace"}},
    "seq_blocks_6": {"config": {"name": "input_only"}},
}

for multiplier in channel_multipliers:
    pass_config = dict(base_pass_config)

    for key in ["seq_blocks_2", "seq_blocks_3", "seq_blocks_4", "seq_blocks_5", "seq_blocks_6"]:
        pass_config[key]["config"]["channel_multiplier"] = multiplier

    print("pass_config:\n", pass_config)
    search_spaces.append(copy.deepcopy(pass_config))



import torch
from torchmetrics.classification import MulticlassAccuracy
metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5

# This first loop is basically our search strategy,
# in this case, it is a simple brute force search
recorded_accs = []


for i, pass_config in enumerate(search_spaces):
    mg_new, _ = redefine_linear_transform_pass(graph=mg, pass_args={"config": pass_config})
    j = 0

    # this is the inner loop, where we also call it as a runner.
    acc_avg, loss_avg = 0, 0
    accs, losses = [], []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs
        preds = mg_new.model(xs)
        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)
        accs.append(acc)
        losses.append(loss)
        if j > num_batchs:
            break
        j += 1
    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    recorded_accs.append(acc_avg)

print("recorded_accs", recorded_accs)
max_acc = max(recorded_accs, key=lambda x: x.item())
print("max_acc", max_acc)
max_acc_index = recorded_accs.index(max_acc)
print("max_acc_index", max_acc_index)
max_acc_search_space = search_spaces[max_acc_index]
print("max_acc_search_space", max_acc_search_space)
max_acc_multiplier = channel_multipliers[max_acc_index]
print("max_acc_multiplier", max_acc_multiplier)












