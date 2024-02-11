# Pracrical 2

## Lab3

### Q1
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


### Q2
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



## Q3

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

## Q4
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

### Q1



### Q2


### Q3


### Q4

