# Pracrical 1

## Lab1

### 1. What is the impact of varying batch sizes and why?

### 2. What is the impact of varying maximum epoch number?
The variation in the maximum number of epochs impacts model training in the following ways:

* **Training Time**: 
Increasing the maximum number of epochs will extend the training duration because the model needs 
to iterate through the entire dataset more times.

* **Risk of Overfitting or Underfitting**: 
If the number of epochs is too high, the model might learn the features of 
the training data excessively such as noise, leading to overfitting. This means the model performs well on the training 
data but poorly on new, unseen data. However, if it is set too low, the model might not have enough time to learn the 
features of the data, leading to underfitting, where the model performs poorly on both training and new data.

* **Convergence**: 
A higher number of epochs may increase the likelihood of the model finding a good or acceptable 
solution. However, continuing to iterate after the model stops improving or even starts overfitting may not bring 
any benefits.

Therefore, the choice of maximum number of epochs should be based on the specific application context, the size and 
complexity of the dataset, and the characteristics of the model such as learning rate.

### 3. What is happening with a large learning and what is happening with a small learning rate and why? What is the relationship between learning rates and batch sizes?
* **Large Learning Rate**: 
It can cause rapid changes in model parameters, thereby accelerating convergence.However, an 
excessively large learning rate might lead to instability in the training process, or even causing the model diverge. 
Another possible issue is that excessively large learning rate may cause the model to skip over the best values during 
the search for an optimal solution because the updates are too large each time.

* **Small Learning Rate**: 
It can make the training process more stable, but it may slow down model convergence, 
requiring more epochs. Meanwhile, an overly small learning rate might cause the model to get stuck in local optima, 
especially in complex loss functions.

* **The relationship between learning rates and batch sizes**: 
Large batch sizes often require higher learning rates because a large batch provides a more stable estimate of the 
gradient, allowing for larger steps in learning. In contrast, small batch sizes might need smaller learning rates as the 
direction of each update could be more volatile, and a smaller learning rate can prevent excessive adjustments.

### 4. Implement a network that has in total around 10x more parameters than the toy network.
As the toy network has 327 trainable parameters, the new network should have around 3270 parameters. Thus, the new 
edited network which has 3.4k parameters is shown in the code snippet below.
```
# modification in mase/machop/chop/models/physical/jet_substructure/__init__.py
class JSC_10x(nn.Module):
    def __init__(self, info):
        super(JSC_10x, self).__init__()
        self.seq_blocks = nn.Sequential(
            # 1st LogicNets Layer
            nn.BatchNorm1d(16),  # input_quant       # 0
            nn.ReLU(16),  # 1
            nn.Linear(16, 60),  # linear              # 2
            nn.BatchNorm1d(60),  # output_quant       # 3
            nn.ReLU(60),  # 4
            # 2nd LogicNets Layer
            nn.Linear(60, 32),  # 5
            nn.BatchNorm1d(32),  # 6
            nn.ReLU(32),  # 7
            # 3rd LogicNets Layer
            nn.Linear(32, 5),  # 8
            nn.BatchNorm1d(5),  # 9
            nn.ReLU(5),
        )

    def forward(self, x):
        return self.seq_blocks(x)
```

The other modifications I made to make it work properly are illustrated:
```
# modification in mase/machop/chop/models/physical/jet_substructure/__init__.py
...
def get_jsc_10x(info):
    return JSC_10x(info)
...
```

```
# modification in mase/machop/chop/models/physical/__init__.py
from .jet_substructure import get_jsc_10x

PHYSICAL_MODELS = {
    ...
    "jsc-10x": {
        "model": get_jsc_10x,
        "info": MaseModelInfo(
            "jsc-10x",
            model_source="physical",
            task_type="physical",
            physical_data_point_classification=True,
            is_fx_traceable=True,
        ),
    },
    ...
}
```

Then by running the command line, the network is trained.
```bash
./ch train jsc-10x jsc --max-epochs 10 --batch-size 256
```

### 5.Test your implementation and evaluate its performance.
The training and validation accuracies derived by running the command in the previous question is:
```
Epoch 9: 100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 3084/3084 [00:16<00:00, 183.06it/s, v_num=0, train_acc_step=0.730, val_acc_epoch=0.700, val_loss_epoch=0.933]
```

Running the test command:
```bash
./ch test jsc-10x jsc --load ../mase_output/jsc-10x_classification_jsc_2024-02-10/software/training_ckpts/best.ckpt --load-type pl
```

The test result is:
```
Testing DataLoader 0: 100%|█████████████████████████████████████████████████████████████████████████| 1542/1542 [00:03<00:00, 469.61it/s]
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
       Test metric             DataLoader 0
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
     test_acc_epoch         0.6988021731376648
     test_loss_epoch        0.9353060722351074
─────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────────
```

Comparing to the jsc-tiny and jsc-toy, the new network achieves much better accuracy. However, due to the significantly 
more trainable parameters, the new network exhibits a tendency of overfitting as the training accuracy is approximately 
3% more than the test accuracy.



## Lab 2

### 1. Explain the functionality of report_graph_analysis_pass and its printed jargons such as placeholder, get_attr … You might find the doc of torch.fx useful.

#### Functionality of `report_graph_analysis_pass` function
The functionality of the `report_graph_analysis_pass` function is to analyse and report detailed information about a 
MaseGraph. It iterates through each node in the graph, counts the number of different types of nodes, and collects the 
types of layers present in the graph. Additionally, if a file name is specified, it writes the report to a file; 
otherwise, it prints the report to the console.

An example of the generated output is shown in the snippet below.
```
graph():
    %x : [num_users=1] = placeholder[target=x]
    %seq_blocks_0 : [num_users=1] = call_module[target=seq_blocks.0](args = (%x,), kwargs = {})
    %seq_blocks_1 : [num_users=1] = call_module[target=seq_blocks.1](args = (%seq_blocks_0,), kwargs = {})
    %seq_blocks_2 : [num_users=1] = call_module[target=seq_blocks.2](args = (%seq_blocks_1,), kwargs = {})
    %seq_blocks_3 : [num_users=1] = call_module[target=seq_blocks.3](args = (%seq_blocks_2,), kwargs = {})
    return seq_blocks_3
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 4, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=5, bias=True), ReLU(inplace=True)]
```

It first illustrates the graph's structure. The subsequent network overview displays the count of different node types. 
Following that, layer types are printed out, listing the specific layer types present in the graph along with their 
configurations. This reveals the layers used in the model and their parameters, including the number of inputs and 
outputs for each layer.

#### Explanation of the printed jargons

* **Placeholder**: 
Represents a node in the graph that stands for input data. It is where the model receives its input.

* **get_attr**: 
It is a function used to retrieve a specified attribute or parameter from a model. It can be used to access weights, 
biases, submodules, or other attributes of the model.

* **call_function**: 
Indicates a node in the graph that performs an operation by calling a function.

* **call_method**: 
Similar to call_function, but here it refers to the invocation of methods on objects, such as operations performed on a 
torch.Tensor object.

* **call_module**: 
Indicates that a node in the graph performs an operation by calling a module. It is an operation used to invoke a 
submodule within a model. It allows to call a submodule as if it were a function, passing inputs and obtaining outputs.

* **output**: 
The output node of the graph, representing the final output of the model.


### 2. What are the functionalities of profile_statistics_analysis_pass and report_node_meta_param_analysis_pass respectively?

#### Functionality of `profile_statistics_analysis_pass`
 
The function takes two main inputs: graph and pass_args. The function then proceeds to perform several operations on the 
graph based on the provided pass_args: 

1. It registers statistics collections for weight and activation profiling on specific nodes in the graph. The 
registration is based on the criteria specified by the `by`, `target_weight_nodes`, and `target_activation_nodes` 
parameters. 

2. It profiles weight statistics by iterating through the graph's nodes and updating statistics collections for weight. 

3. It profiles activation statistics by iterating through the input data using the `graph_iterator_profile_act` function. 
This involves running the graph with input data and updating activation statistics collections.

4. Finally, it computes and unregisters the collected statistics.

The function returns the modified graph and an empty dictionary as the result of the profile statistics analysis.

#### Functionality of `report_node_meta_param_analysis_pass`

It takes the graph and pass_args as inputs. It first selectively generates a report table based on the meta-parameter 
types specified by the "which" parameter. The function iterates through the nodes of the graph, selectively includes 
meta-parameter information based on the "which" parameter, formats this information into rows, and adds them to the 
report table.
 
Within the table, the software parameters of each node contain different statistic information such as variance and 
mean. This information is collected via the function `profile_statistics_analysis_pass`. The software parameters differ 
due to the analysed node types and configurations, e.g., some node contains the statistic information of activation 
range and others may contain the statistic information of the variance of the weights.

### 3. Explain why only 1 OP is changed after the quantize_transform_pass .

This is because the input pass_args of the function `quantize_transform_pass()` defines "linear", which makes the function 
only consider and transform the node of linear layer. The other nodes are not listed in pass_args and thus will not be 
transformed due to the if statement in function `graph_iterator_quantize_by_type()`.

### 4. Write some code to traverse both mg and ori_mg, check and comment on the nodes in these two graphs. You might find the source code for the implementation of summarize_quantization_analysis_pass useful.
The code snippet below shows how to traverse both `mg` and `ori_mg`. `summarize_quantization_analysis_pass` is used to 
traverse and compare between the two graph. To illustrate intuitively how the node is transformed, the function 
`graph_iterator_compare_nodes` in `/chop/passes/graph/transforms/quantize/summary` is used, which generates a dataframe 
that records the nodes and the corresponding configurations.
```
import pandas as pd
from chop.passes.graph.transforms.quantize.summary import graph_iterator_compare_nodes

...
mg, _ = quantize_transform_pass(mg, pass_args1)
summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")

df = graph_iterator_compare_nodes(ori_mg, mg, save_path=None, silent=False)
pd.set_option('display.max_columns', None)
print(df)
...
```

The result is demonstrated as follows. It is clear that the node Linear is transformed to LinearInteger, whereas the 
nodes remains the same.
```
       Ori name      New name            MASE_TYPE       Mase_OP   Original type Quantized type  Changed  
0             x             x          placeholder   placeholder               x              x    False
1  seq_blocks_0  seq_blocks_0               module  batch_norm1d     BatchNorm1d    BatchNorm1d    False
2  seq_blocks_1  seq_blocks_1  module_related_func          relu            ReLU           ReLU    False  
3  seq_blocks_2  seq_blocks_2  module_related_func        linear          Linear  LinearInteger     True  
4  seq_blocks_3  seq_blocks_3  module_related_func          relu            ReLU           ReLU    False
5        output        output               output        output          output         output    False  
```

### 5. Perform the same quantisation flow to the bigger JSC network that you have trained in lab1. You must be aware that now the pass_args for your custom network might be different if you have used more than the Linear layer in your network.

Similar to the previous quantisation flow, the quantisation follow for the new network is:

```
batch_size = 8
model_name = "jsc-10x"
dataset_name = "jsc"


data_module = MaseDataModule(
    name=dataset_name,
    batch_size=batch_size,
    model_name=model_name,
    num_workers=0,
)
data_module.prepare_data()
data_module.setup()

CHECKPOINT_PATH = "/home/lijun/mase/mase_output/jsc-10x_classification_jsc_2024-02-10/software/training_ckpts/best.ckpt"
model_info = get_model_info(model_name)
model = get_model(
    model_name,
    task="cls",
    dataset_info=data_module.dataset_info,
    pretrained=False)

model = load_model(load_name=CHECKPOINT_PATH, load_type="pl", model=model)

input_generator = InputGenerator(
    data_module=data_module,
    model_info=model_info,
    task="cls",
    which_dataloader="train",
)

# a demonstration of how to feed an input value to the model
dummy_in = next(iter(input_generator))
_ = model(**dummy_in)

mg = MaseGraph(model=model)
mg, _ = init_metadata_analysis_pass(mg, None)
mg, _ = add_common_metadata_analysis_pass(mg, {"dummy_in": dummy_in})
mg, _ = add_software_metadata_analysis_pass(mg, None)
from chop.passes.graph import report_graph_analysis_pass
_ = report_graph_analysis_pass(mg)

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

mg, _ = profile_statistics_analysis_pass(mg, pass_args)
mg, _ = report_node_meta_param_analysis_pass(mg, {"which": ("software",)})

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

ori_mg = MaseGraph(model=model)
ori_mg, _ = init_metadata_analysis_pass(ori_mg, None)
ori_mg, _ = add_common_metadata_analysis_pass(ori_mg, {"dummy_in": dummy_in})

mg, _ = quantize_transform_pass(mg, pass_args1)
summarize_quantization_analysis_pass(ori_mg, mg, save_dir="quantize_summary")
```
The generated result is illustrated below:
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
    %seq_blocks_8 : [num_users=1] = call_module[target=seq_blocks.8](args = (%seq_blocks_7,), kwargs = {})
    %seq_blocks_9 : [num_users=1] = call_module[target=seq_blocks.9](args = (%seq_blocks_8,), kwargs = {})
    %seq_blocks_10 : [num_users=1] = call_module[target=seq_blocks.10](args = (%seq_blocks_9,), kwargs = {})
    return seq_blocks_10
Network overview:
{'placeholder': 1, 'get_attr': 0, 'call_function': 0, 'call_method': 0, 'call_module': 11, 'output': 1}
Layer types:
[BatchNorm1d(16, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=16, out_features=60, bias=True), BatchNorm1d(60, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=60, out_features=32, bias=True), BatchNorm1d(32, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True), Linear(in_features=32, out_features=5, bias=True), BatchNorm1d(5, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True), ReLU(inplace=True)]
INFO     Inspecting graph [add_common_meta_param_analysis_pass]
INFO     
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| Node name     | Fx Node op   | Mase type           | Mase op      | Software Param                                                                           |
+===============+==============+=====================+==============+==========================================================================================+
| x             | placeholder  | placeholder         | placeholder  | {'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_0  | call_module  | module              | batch_norm1d | {'args': {'bias': {'stat': {}},                                                          |
|               |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|               |              |                     |              |           'running_mean': {'stat': {}},                                                  |
|               |              |                     |              |           'running_var': {'stat': {}},                                                   |
|               |              |                     |              |           'weight': {'stat': {}}},                                                       |
|               |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_1  | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 512,                        |
|               |              |                     |              |                                                     'max': 1.8981868028640747,           |
|               |              |                     |              |                                                     'min': -1.4900847673416138,          |
|               |              |                     |              |                                                     'range': 3.3882715702056885}}}},     |
|               |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_2  | call_module  | module_related_func | linear       | {'args': {'bias': {'stat': {'variance_precise': {'count': 60,                            |
|               |              |                     |              |                                                  'mean': -0.005789244547486305,          |
|               |              |                     |              |                                                  'variance': 0.019366875290870667}}},    |
|               |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|               |              |                     |              |           'weight': {'stat': {'variance_precise': {'count': 960,                         |
|               |              |                     |              |                                                    'mean': -0.0024341046810150146,       |
|               |              |                     |              |                                                    'variance': 0.021727358922362328}}}}, |
|               |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_3  | call_module  | module              | batch_norm1d | {'args': {'bias': {'stat': {}},                                                          |
|               |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|               |              |                     |              |           'running_mean': {'stat': {}},                                                  |
|               |              |                     |              |           'running_var': {'stat': {}},                                                   |
|               |              |                     |              |           'weight': {'stat': {}}},                                                       |
|               |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_4  | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 1920,                       |
|               |              |                     |              |                                                     'max': 1.9613263607025146,           |
|               |              |                     |              |                                                     'min': -2.034088611602783,           |
|               |              |                     |              |                                                     'range': 3.995414972305298}}}},      |
|               |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_5  | call_module  | module_related_func | linear       | {'args': {'bias': {'stat': {'variance_precise': {'count': 32,                            |
|               |              |                     |              |                                                  'mean': 0.01643959991633892,            |
|               |              |                     |              |                                                  'variance': 0.004667417611926794}}},    |
|               |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|               |              |                     |              |           'weight': {'stat': {'variance_precise': {'count': 1920,                        |
|               |              |                     |              |                                                    'mean': -0.004181257449090481,        |
|               |              |                     |              |                                                    'variance': 0.005490461830049753}}}}, |
|               |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_6  | call_module  | module              | batch_norm1d | {'args': {'bias': {'stat': {}},                                                          |
|               |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|               |              |                     |              |           'running_mean': {'stat': {}},                                                  |
|               |              |                     |              |           'running_var': {'stat': {}},                                                   |
|               |              |                     |              |           'weight': {'stat': {}}},                                                       |
|               |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_7  | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 1024,                       |
|               |              |                     |              |                                                     'max': 1.7301989793777466,           |
|               |              |                     |              |                                                     'min': -2.0961928367614746,          |
|               |              |                     |              |                                                     'range': 3.8263916969299316}}}},     |
|               |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_8  | call_module  | module_related_func | linear       | {'args': {'bias': {'stat': {'variance_precise': {'count': 5,                             |
|               |              |                     |              |                                                  'mean': 0.020693045109510422,           |
|               |              |                     |              |                                                  'variance': 0.015003956854343414}}},    |
|               |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|               |              |                     |              |           'weight': {'stat': {'variance_precise': {'count': 160,                         |
|               |              |                     |              |                                                    'mean': 0.0020955046638846397,        |
|               |              |                     |              |                                                    'variance': 0.009414528496563435}}}}, |
|               |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_9  | call_module  | module              | batch_norm1d | {'args': {'bias': {'stat': {}},                                                          |
|               |              |                     |              |           'data_in_0': {'stat': {}},                                                     |
|               |              |                     |              |           'running_mean': {'stat': {}},                                                  |
|               |              |                     |              |           'running_var': {'stat': {}},                                                   |
|               |              |                     |              |           'weight': {'stat': {}}},                                                       |
|               |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| seq_blocks_10 | call_module  | module_related_func | relu         | {'args': {'data_in_0': {'stat': {'range_quantile': {'count': 160,                        |
|               |              |                     |              |                                                     'max': 2.245518207550049,            |
|               |              |                     |              |                                                     'min': -1.6693905591964722,          |
|               |              |                     |              |                                                     'range': 3.9149088859558105}}}},     |
|               |              |                     |              |  'results': {'data_out_0': {'stat': {}}}}                                                |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
| output        | output       | output              | output       | {'args': {'data_in_0': {'stat': {}}}}                                                    |
+---------------+--------------+---------------------+--------------+------------------------------------------------------------------------------------------+
INFO     Quantized graph histogram:
INFO     
| Original type   | OP           |   Total |   Changed |   Unchanged |
|-----------------+--------------+---------+-----------+-------------|
| BatchNorm1d     | batch_norm1d |       4 |         0 |           4 |
| Linear          | linear       |       3 |         3 |           0 |
| ReLU            | relu         |       4 |         0 |           4 |
| output          | output       |       1 |         0 |           1 |
| x               | placeholder  |       1 |         0 |           1 |

```

As can be seen, the three linear layers are changed, transforming from Linear to LinearInteger. This result proves that
the quantisation flow is successful.

### 6. Write code to show and verify that the weights of these layers are indeed quantised. You might need to go through the source code of the implementation of the quantisation pass and also the implementation of the Quantized Layers .

The code snippet is shown below.
```
import torch
from chop.passes.graph.utils import get_node_actual_target

for ori_n, quan_n in zip(ori_mg.fx_graph.nodes, mg.fx_graph.nodes):
    ori_target = get_node_actual_target(ori_n)
    quan_target = get_node_actual_target(quan_n)

    if type(ori_target) != type(quan_target):
        print(ori_target, quan_target)
        print("Weights of original graph:")
        print(ori_target.weight)
        print("Weights of the transformed graph:")
        print(quan_target.weight)

        random_input = torch.randn(quan_target.in_features)
        print("random_input", random_input)
        ori_output = ori_target(random_input)
        print("ori_output", ori_output)
        quan_output = quan_target(random_input)
        print("quan_output", quan_output)
```

The generated result from the above code is:
```
Original graph target: Linear(in_features=16, out_features=5, bias=True)

Quantised graph target: LinearInteger(in_features=16, out_features=5, bias=True)

Weights of original graph: Parameter containing:
tensor([[-0.0689,  0.2381, -0.1237, -0.1131, -0.0119,  0.1405,  0.0843,  0.2918,
          0.0660,  0.1715,  0.0312,  0.0546, -0.1353, -0.0641, -0.0154,  0.1146],
        [ 0.1873,  0.0637, -0.2517, -0.1785,  0.0107,  0.1331, -0.0340,  0.2173,
         -0.0231,  0.0446,  0.2467, -0.1569, -0.1384,  0.0051, -0.1813,  0.1317],
        [-0.0786, -0.0300, -0.2051, -0.3175, -0.1980,  0.1422,  0.1053,  0.0612,
          0.0046, -0.1761,  0.0433, -0.2522, -0.2328, -0.1842,  0.1206,  0.0641],
        [-0.1261, -0.0709,  0.0943,  0.1820,  0.0368, -0.0264,  0.1351, -0.1875,
          0.0157, -0.2376, -0.2172, -0.1702,  0.0714,  0.0563, -0.2126,  0.0153],
        [ 0.0501,  0.0420,  0.0959,  0.1444,  0.2406,  0.3258, -0.2193, -0.1061,
          0.0770,  0.1298,  0.1352,  0.1425, -0.0259, -0.2731,  0.1075, -0.0879]],
       requires_grad=True)

Weights of the quantised graph: Parameter containing:
tensor([[-0.0689,  0.2381, -0.1237, -0.1131, -0.0119,  0.1405,  0.0843,  0.2918,
          0.0660,  0.1715,  0.0312,  0.0546, -0.1353, -0.0641, -0.0154,  0.1146],
        [ 0.1873,  0.0637, -0.2517, -0.1785,  0.0107,  0.1331, -0.0340,  0.2173,
         -0.0231,  0.0446,  0.2467, -0.1569, -0.1384,  0.0051, -0.1813,  0.1317],
        [-0.0786, -0.0300, -0.2051, -0.3175, -0.1980,  0.1422,  0.1053,  0.0612,
          0.0046, -0.1761,  0.0433, -0.2522, -0.2328, -0.1842,  0.1206,  0.0641],
        [-0.1261, -0.0709,  0.0943,  0.1820,  0.0368, -0.0264,  0.1351, -0.1875,
          0.0157, -0.2376, -0.2172, -0.1702,  0.0714,  0.0563, -0.2126,  0.0153],
        [ 0.0501,  0.0420,  0.0959,  0.1444,  0.2406,  0.3258, -0.2193, -0.1061,
          0.0770,  0.1298,  0.1352,  0.1425, -0.0259, -0.2731,  0.1075, -0.0879]],
       requires_grad=True)

Random input for the graph:
 tensor([ 0.5319,  0.2876, -0.7508, -0.2564, -0.8496, -0.1951, -0.7838,  0.7940,
        -0.2874,  0.8280, -0.1643, -0.7469,  0.3893,  0.3645, -0.7502, -0.3598])

Output of the original graph:
 tensor([ 0.1412,  0.8343,  0.3334, -0.6339, -0.4795], grad_fn=<ViewBackward0>)

Output of the quantised graph:
 tensor([ 0.1914,  0.9180,  0.2656, -0.6914, -0.4453], grad_fn=<ViewBackward0>)
```


