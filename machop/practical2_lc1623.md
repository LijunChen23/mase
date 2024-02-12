# Pracrical 2

## Lab3

### 1. Explore additional metrics that can serve as quality metrics for the search process. For example, you can consider metrics such as latency, model size, or the number of FLOPs (floating-point operations) involved in the model.
Exploring additional metrics to serve as quality indicators during the search process can significantly enhance the 
evaluation and selection of models. Here, we discuss three such metrics: latency, model size, and the number of FLOPs.

* **Latency**:
Latency measures the time it takes for a model to make a prediction after receiving input. It's crucial for real-time 
applications where quick decision-making is essential. Optimizing for latency ensures that the model can operate 
effectively within the time constraints of its intended application.

* **Model Size**:
The model size refers to the amount of memory required to store the model's parameters. It's a critical factor for 
deploying models on devices with limited storage capacity. Smaller models are easier to deploy in resource-constrained 
environments. Reducing model size can also decrease load times and memory usage, improving overall application 
performance.

* **Number of FLOPs**:
FLOPs indicate the computational complexity of a model. It measures the number of floating-point calculations the model 
performs during inference. Optimizing for FLOPs can lead to models that balance performance with resource utilization.


### 2. Implement some of these additional metrics and attempt to combine them with the accuracy or loss quality metric. It’s important to note that in this particular case, accuracy and loss actually serve as the same quality metric (do you know why?).
#### Implementation of combining latency and model size with the accuracy metric
Latency is measured by recording the time before and after the model makes predictions on the input data, thus capturing 
the time taken for each prediction. The model size is evaluated by calculating the total number of parameters in the 
model, which serves as an indicator of the model's complexity and memory requirements. By integrating these metrics, the 
model performance can be assessed more comprehensively, taking into account not only the accuracy but also the 
efficiency and resource utilization of the model.
```
import time


metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5

recorded_accs = []
recorded_latencies = []
recorded_model_sizes = []

for i, config in enumerate(search_spaces):
    mg, _ = quantize_transform_pass(mg, config)
    j = 0

    acc_avg, loss_avg, latency_avg = 0, 0, 0
    accs, losses, latencies = [], [], []
    for inputs in data_module.train_dataloader():
        xs, ys = inputs

        start_time = time.time()
        preds = mg.model(xs)
        elapsed_time = time.time() - start_time

        loss = torch.nn.functional.cross_entropy(preds, ys)
        acc = metric(preds, ys)

        accs.append(acc)
        losses.append(loss)
        latencies.append(elapsed_time)

        if j > num_batchs:
            break
        j += 1

    acc_avg = sum(accs) / len(accs)
    loss_avg = sum(losses) / len(losses)
    latency_avg = sum(latencies) / len(latencies)
    total_params = sum(p.numel() for p in mg.model.parameters())

    recorded_accs.append(acc_avg)
    recorded_latencies.append(latency_avg)
    recorded_model_sizes.append(total_params)

# ...later works to determine which models to use according to the recorded accuracies, latencies and model sizes.
```


#### Why do accuracy and loss actually serve as the same quality metric?
Accuracy measures the proportion of correct predictions made by the model over the total number of predictions. 
The loss provides a more granular view of the model's performance, indicating not just whether the model is right or 
wrong, but how far off its predictions are. In the context of classification tasks, accuracy and loss often move in 
tandem, reflecting the model's learning progression and effectiveness in making correct predictions.



### 3. Implement the brute-force search as an additional search method within the system, this would be a new search strategy in MASE.

As brute-force search method is integrated in optuna library, we can simply add an extra case "brute-force" under the 
`/chop/actions/search/strategies/optuna.py`which is shown in the code snippet below.

```
def sampler_map(self, name):
    match name.lower():
        ...rest of the code remains unchanged
        # Added case
        case "brute-force":
            sampler = optuna.samplers.BruteForceSampler()
        ...
    return sampler
```

Meanwhile, `the jsc_toy_by_type.toml` should also be adjusted to fit the brute-force search strategy. The adjustment is 
shown as follows. The sampler "tpe" is changed to "brute-force".

```
[search.strategy.setup]
...
#sampler = "tpe"
sampler = "brute-force"
...
```

### 4. Compare the brute-force search with the TPE based search, in terms of sample efficiency. Comment on the performance difference between the two search methods.
Brute-force search seeks the optimal solution by enumerating all possible configuration combinations, meaning it 
requires evaluating a vast number of configurations, leading to lower sample efficiency. In contrast, TPE as a 
model-based search strategy, intelligently selects the next set of parameters to evaluate by learning from the results 
of past trials, thereby improving sample efficiency. TPE aims to find the optimal solution with as few trials as 
possible. In summary, although brute-force search can theoretically guarantee the discovery of the optimal solution, its 
sample efficiency is lower due to the need to evaluate every possible configuration in the search space. Meanwhile, TPE 
improves sample efficiency by guiding the search based on the results of historical trials, allowing it to find good 
solutions within a limited number of configuration evaluations, even though this may not guarantee a global optimum.


The performance of TPE search method is:
```
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        2 | {'loss': 1.47, 'accuracy': 0.414}  | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.414, 'average_bitwidth': 0.4} |
|  1 |        3 | {'loss': 1.468, 'accuracy': 0.416} | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.416, 'average_bitwidth': 1.6} |


Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        2 | {'loss': 1.459, 'accuracy': 0.439} | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.439, 'average_bitwidth': 1.6} |
|  1 |        3 | {'loss': 1.457, 'accuracy': 0.422} | {'average_bitwidth': 4.0, 'memory_density': 8.0}  | {'accuracy': 0.422, 'average_bitwidth': 0.8} |
|  2 |       11 | {'loss': 1.484, 'accuracy': 0.42}  | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.42, 'average_bitwidth': 0.4}  |
```

The performance of brute-force search method is:
```
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        3 | {'loss': 1.496, 'accuracy': 0.416} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.416, 'average_bitwidth': 0.4} |
|  1 |        4 | {'loss': 1.485, 'accuracy': 0.422} | {'average_bitwidth': 8.0, 'memory_density': 4.0}  | {'accuracy': 0.422, 'average_bitwidth': 1.6} |


INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                  | scaled_metrics                               |
|----+----------+------------------------------------+---------------------------------------------------+----------------------------------------------|
|  0 |        0 | {'loss': 1.44, 'accuracy': 0.451}  | {'average_bitwidth': 16.0, 'memory_density': 2.0} | {'accuracy': 0.451, 'average_bitwidth': 3.2} |
|  1 |        2 | {'loss': 1.491, 'accuracy': 0.426} | {'average_bitwidth': 2.0, 'memory_density': 16.0} | {'accuracy': 0.426, 'average_bitwidth': 0.4} |
```

In the results provided, the best configuration found using TPE search achieved an accuracy of 0.416. The best 
configuration found by brute-force search achieved an accuracy of 0.422, which is the outcome after traversing all 
configurations.  The results indicate that brute-force theoretically achieves better results in terms of accuracy, but
TPE approached the performance of brute-force search with fewer configuration evaluations.



## Lab 4

### 1. Can you edit your code, so that we can modify the above network to have layers expanded to double their sizes? Note: you will have to change the ReLU also.

The modified code introduces processing for nn.ReLU modules. In the original code, the focus was primarily on adjusting 
the input and output feature sizes of nn.Linear modules. In the modified version, if a ReLU module is encountered, the 
inplace attribute of the ReLU module is adjusted based on the channel_multiplier, which is a newly added functionality.
However, one thing to notice is that even though the input `Inplace` of ReLU is modified, it is actually an unnecessary 
action because whatever the input number is, the ReLU considers it as `True`. However, such modification at least boosts 
the understanding of feature size of each layer.

```
def instantiate_ReLU(inplace):
    return nn.ReLU(inplace=inplace)

def redefine_linear_transform_pass(graph, pass_args=None):
    main_config = pass_args.pop('config')
    default = main_config.pop('default', None)
    if default is None:
        raise ValueError(f"default value must be provided.")
    i = 0
    for node in graph.fx_graph.nodes:
        i += 1
        # if node name is not matched, it won't be tracked
        config = main_config.get(node.name, default)['config']
        name = config.get("name", None)

        if name is not None:
            ori_module = graph.modules[node.target]
            if isinstance(ori_module, nn.ReLU):
                inplace = ori_module.inplace
                if name == "inplace":
                    inplace = inplace * config["channel_multiplier"]
                new_module = instantiate_ReLU(inplace)
            elif isinstance(ori_module, nn.Linear):
                in_features = ori_module.in_features
                out_features = ori_module.out_features
                bias = ori_module.bias
                if name == "output_only":
                    out_features = out_features * config["channel_multiplier"]
                elif name == "both":
                    in_features = in_features * config["channel_multiplier"]
                    out_features = out_features * config["channel_multiplier"]
                elif name == "input_only":
                    in_features = in_features * config["channel_multiplier"]
                new_module = instantiate_linear(in_features, out_features, bias)
            parent_name, name = get_parent_name(node.target)
            setattr(graph.modules[parent_name], name, new_module)
    return graph, {}
```

Additionally, the `pass_config` is also modified, adding configurations for the ReLU layer.
```
pass_config = {
    "by": "name",
    "default": {"config": {"name": None}},
    "seq_blocks_2": {
        "config": {
            "name": "output_only",
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
```


### 2. In lab3, we have implemented a grid search, can we use the grid search to search for the best channel multiplier value?

Initially, the code below defines a base configuration dictionary `base_pass_config`, specifying how different 
configuration strategies should be applied to specific sequence blocks by name. Then by iterating over different 
channel multipliers, it generates configuration variants for each sequence block. These variants will be used to 
evaluate different model configurations during the model search process.
```
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
```

After that, the implementation of grid search is achieved using the code snippet shown as follows. Using the defined 
search spaces, each configuration variant undergoes model transformation and evaluation. The transformation function 
`redefine_linear_transform_pass` adjusts the feature sizes of specified layers according to `pass_config`. For each 
channel multiplier, the model's performance is evaluated by running it on training data and calculating average 
accuracy and loss. The `MulticlassAccuracy` from the `torchmetrics` library is used to compute accuracy. Finally, the 
code compares the average accuracy of different channel multiplier and selects the one with the best performance. It 
also outputs the corresponding channel multiplier, providing insights into which configuration achieved the best
performance.

```
import torch
from torchmetrics.classification import MulticlassAccuracy
metric = MulticlassAccuracy(num_classes=5)
num_batchs = 5

# This first loop is basically our search strategy, in this case, it is a simple brute force search
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

print("Recorded accuracies", recorded_accs)
max_acc = max(recorded_accs, key=lambda x: x.item())
print("Maximum accuracy: ", max_acc)
max_acc_index = recorded_accs.index(max_acc)
max_acc_multiplier = channel_multipliers[max_acc_index]
print("Corresponding channel multiplier for this accuracy: ", max_acc_multiplier)
```

By running the code above, the printed result is shown below. This shows that among channel multiplier values of 2, 4, 
and 8, the transformed model achieved the highest accuracy when the channel multiplier was set to 2.
```
Recorded accuracies [tensor(0.2381), tensor(0.1452), tensor(0.2167)]
Maximum accuracy:  tensor(0.2381)
Corresponding channel multiplier for this accuracy:  2
```

### 3. Can you then design a search so that it can reach a network that can have this kind of structure?
The modified `redefine_linear_transform_pass` function is shown in the following code snippet. The main change here 
involves the handling of the "both" type. In the original code, the "both" type only supported a single 
`channel_multiplier` for adjusting both input and output feature numbers simultaneously. However, in the modified code, 
`channel_multiplier_input` and `channel_multiplier_output` are introduced as two separate configuration items, allowing 
for different scaling ratios for input and output channels, respectively.

```
def redefine_linear_transform_pass(graph, pass_args=None):
    ...
    for node in graph.fx_graph.nodes:  # node.name = e.g. seq_blocks_2
        ...
        if name is not None:
            ori_module = graph.modules[node.target]  # e.g., node.target = "seq_blocks.4"
            if isinstance(ori_module, nn.ReLU):
                ...
            elif isinstance(ori_module, nn.Linear):
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
            ...
    return graph, {}
```

Additionally, the pass_config should also be modified to allow two separate configuration items of the "both" type. This 
modification is:
```
pass_config = {
    "by": "name",
    "default": {"config": {"name": None}},
    "seq_blocks_2": {
        "config": {
            "name": "output_only",
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
            "channel_multiplier_input": 2,
            "channel_multiplier_output": 4,
        }
    },
    "seq_blocks_5": {
        "config": {
            "name": "inplace",
            "channel_multiplier": 4,
        }
    },
    "seq_blocks_6": {
        "config": {
            "name": "input_only",
            "channel_multiplier": 4,
        }
    },
}
```

Finally, we can run the function `redefine_linear_transform_pass()` to transform the graph and then report the 
transformed graph analysis to check whether the transformed channels are as desired.
```
from chop.passes.graph import report_graph_analysis_pass
# this performs the architecture transformation based on the config
mg, _ = redefine_linear_transform_pass(graph=mg, pass_args={"config": pass_config})
_ = report_graph_analysis_pass(mg)
```
The printout of the above code is as follows, showing that the separate channel multiplication is successful.
```
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=32, bias=True), ReLU(inplace=True), Linear(in_features=32, out_features=64, bias=True), ReLU(inplace=True), Linear(in_features=64, out_features=5, bias=True), ReLU(inplace=True)]
```


### 4. Integrate the search to the chop flow, so we can run it from the command line.

First, a Python package `/mase/machop/chop/actions/search/search_space/channel_multiplication` is created which is 
similar to the package `mase/machop/chop/actions/search/search_space/quantization`. Two Python files are created inside:
`__init__.py` and `graph.py`. 

The code in `__init__.py` is:
```
from .graph import GraphSearchSpaceChannelMultiplier
```

The code in `graph.py` under `channel_multiplication` package is a duplicate to `graph.py` under `quantization` package. 
However, the below code snippet only shows the modifications of the `graph.py` code. Within the function 
`rebuild_model()`, the function `redefine_linear_transform_pass()` same as question 3 is utilised. Note that 
`report_graph_analysis_pass()` function is imported and utilised to verify whether the graph is transformed successfully.
```
...

from .....passes.graph.transforms.channel_multiply import (
    QUANTIZEABLE_OP,
    redefine_linear_transform_pass,
)

from .....passes.graph import report_graph_analysis_pass
...

DEFAULT_CHANNEL_MULTIPLICATION_CONFIG = {
    "config": {"name": None}
}


class GraphSearchSpaceChannelMultiplier(SearchSpaceBase):
    """
    Post-Training channel multiplier search space for mase graph.
    """

    def _post_init_setup(self):
        ...
        
        self.default_config = DEFAULT_CHANNEL_MULTIPLICATION_CONFIG
        
        ...

    def rebuild_model(self, sampled_config, is_eval_mode: bool = True):
    
        ...
        
        if sampled_config is not None:
            """
            Modified function
            """
            mg, _ = redefine_linear_transform_pass(self.mg, pass_args={"config": sampled_config})
        mg.model.to(self.accelerator)
        # Verify whether the graph is transformed successfully.
        _ = report_graph_analysis_pass(mg)
        return mg

    ...
    # The rest are almost the same, where only several print() functions are used to examine the variables,
    # hence, it is not necessary to exhibit the all of the rest of the code.
    ...
    
    def flattened_indexes_to_config(self, indexes: dict[str, int]):
        ...
        print("\nconfig:\n", config)
        return config
```

Modifications are also made in `/mase/machop/chop/actions/search/search_space/__init__.py`:
```
...
from .channel_multiplication import GraphSearchSpaceChannelMultiplier
...

SEARCH_SPACE_MAP = {
    ...
    "graph/channel_multiply/channel_multiplier": GraphSearchSpaceChannelMultiplier,
    ...
}
...
```

In addition, another Python package `/mase/machop/chop/passes/graph/transforms/channel_multiply` is created, which is 
similar to package `/mase/machop/chop/passes/graph/transforms/quantize`. Under this package, two files are created: 
`__init__.py` and `channel_multiply.py`.

The code in `__init__.py` is:
```
from .channel_multiply import QUANTIZEABLE_OP, redefine_linear_transform_pass
```

The code in `channel_multiply.py` under `channel_multiply` package is a duplicate to `quantize.py` under `quantize` 
package. The below code snippet only shows the modifications of the `channel_multiply.py` code. The function 
`redefine_linear_transform_pass()` same as question 3 is defined in this file. And the `quantize_transform_pass()` 
function is removed.

```
...

from ..quantize.modify import create_new_fn, create_new_module
from ..quantize.quant_parsers import parse_node_config, relink_node_meta, update_quant_meta_param

...

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
```

The above illustrates all the Python packages and files needed to perform the chop flow of channel multiplication. 
Eventually, a `.toml` file is created namely `/mase/machop/configs/examples/jsc_three_linear_layers_by_type.toml`. It is 
a duplicate of `/mase/machop/configs/examples/jsc_toy_by_type.toml`. The below code snippet will only demonstrate the 
changes made. The configurations of the sequence blocks 2, 3, 4, 5, and 6 are specified similarly as question 3.
```
# basics
model = "jsc-three-linear-layers"
dataset = "jsc"
task = "cls"

max_epochs = 5
batch_size = 512
learning_rate = 1e-2
accelerator = "gpu"
project = "jsc-three-linear-layers"
seed = 42
log_every_n_steps = 5
load_name = "/home/lijun/mase/mase_output/jsc-three-linear-layers_classification_jsc_2024-02-09/software/training_ckpts/best.ckpt"
load_type = "pl"

[passes.quantize]
by = "type"
[passes.quantize.default.config]
name = "NA"

[search.search_space]
name = "graph/channel_multiply/channel_multiplier"

[search.search_space.setup]
by = "name"

[search.search_space.seed.default.config]
# the only choice "NA" is used to indicate that layers are not quantized by default
name = ["NA"]

[search.search_space.seed.seq_blocks_2.config]
# if search.search_space.setup.by = "name", this seed will be used to quantize the mase graph node with name "seq_blocks_2"
name = ["output_only"]
channel_multiplier = [2]

[search.search_space.seed.seq_blocks_3.config]
# if search.search_space.setup.by = "name", this seed will be used to quantize the mase graph node with name "seq_blocks_2"
name = ["inplace"]
channel_multiplier = [2]

[search.search_space.seed.seq_blocks_4.config]
# if search.search_space.setup.by = "name", this seed will be used to quantize the mase graph node with name "seq_blocks_2"
name = ["both"]
channel_multiplier_input = [2]
channel_multiplier_output = [4]

[search.search_space.seed.seq_blocks_5.config]
# if search.search_space.setup.by = "name", this seed will be used to quantize the mase graph node with name "seq_blocks_2"
name = ["inplace"]
channel_multiplier = [2]

[search.search_space.seed.seq_blocks_6.config]
# if search.search_space.setup.by = "name", this seed will be used to quantize the mase graph node with name "seq_blocks_2"
name = ["input_only"]
channel_multiplier = [4]


[search.strategy]
...

[search.strategy.sw_runner.basic_evaluation]
...

[search.strategy.hw_runner.average_bitwidth]
...

[search.strategy.setup]
...
n_trials = 1
...
sampler = "brute-force"
...

[search.strategy.metrics]
...
```

With all the code edited, the `chop` flow can be performed using the command line below:
```
./ch search --config configs/examples/jsc_three_linear_layers_by_type.toml
```

The generated result is as follows. The printed ‘config’ is a printout I made to monitor the pass configuration for the 
transformation process. Due to the `report_graph_analysis_pass()` function I added, it is obvious that the search flow 
for the channel multiplier is successful.
```
config:
 {'seq_blocks_1': {'config': {'name': None}}, 'seq_blocks_2': {'config': {'name': 'output_only', 'channel_multiplier': 2}}, 'seq_blocks_3': {'config': {'name': 'inplace', 'channel_multiplier': 2}}, 'seq_blocks_4': {'config': {'name': 'both', 'channel_multiplier_input': 2, 'channel_multiplier_output': 4}}, 'seq_blocks_5': {'config': {'name': 'inplace', 'channel_multiplier': 2}}, 'seq_blocks_6': {'config': {'name': 'input_only', 'channel_multiplier': 4}}, 'seq_blocks_7': {'config': {'name': None}}, 'default': {'config': {'name': None}}, 'by': 'name'}
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    %seq_blocks_4 : [num_users=1] = call_module[target=seq_blocks.4](args = (%seq_blocks_3,), kwargs = {})
    %seq_blocks_5 : [num_users=1] = call_module[target=seq_blocks.5](args = (%seq_blocks_4,), kwargs = {})
    %seq_blocks_6 : [num_users=1] = call_module[target=seq_blocks.6](args = (%seq_blocks_5,), kwargs = {})
    %seq_blocks_7 : [num_users=1] = call_module[target=seq_blocks.7](args = (%seq_blocks_6,), kwargs = {})
    return seq_blocks_7
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 8, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=32, bias=True), ReLU(inplace=True), Linear(in_features=32, out_features=64, bias=True), ReLU(inplace=True), Linear(in_features=64, out_features=5, bias=True), ReLU(inplace=True)]
WARNING  No quantized layers found in the model, set average_bitwidth to 32
100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 1/1 [00:00<00:00,  1.25it/s, 0.80/20000 seconds]
INFO     Best trial(s):
Best trial(s):
|    |   number | software_metrics                   | hardware_metrics                                | scaled_metrics                               |
|----+----------+------------------------------------+-------------------------------------------------+----------------------------------------------|
|  0 |        0 | {'loss': 1.612, 'accuracy': 0.104} | {'average_bitwidth': 32, 'memory_density': 1.0} | {'accuracy': 0.104, 'average_bitwidth': 6.4} |
INFO     Searching is completed
```



