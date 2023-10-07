# <b><u>Tensorflow</u></b>

## Hyperparameter Tuning

---

#### <u>[1] Regression</u>

| Hyperparameter Name              | Allowable Values                                                                                                                                                                         |
|:---------------------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `activation_hidden`              | - `ReLU`<br/>- `ReLU_Leaky`<br/>- `ELU`<br/>- `SELU`<br/>- `PReLU`<br/>- `Tanh`<br/>- `Sigmoid`<br/>- `Softplus`<br/>- `Swish`                                                           |
| `activation_output`              | - `ReLU`<br/>- `ReLU_Leaky`<br/>- `ELU`<br/>- `SELU`<br/>- `PReLU`<br/>- `Tanh`<br/>- `Sigmoid`<br/>- `Softplus`<br/>- `Swish`                                                           |
| `layer_shape_input`              | Any `int`values that are at least `1`                                                                                                                                                    |
| `layer_shape_output`             | Any `int`values that are at least `1`                                                                                                                                                    |
| `loss_function`                  | - `MSE`<br/>- `MAE`<br/>- `Huber`<br/>- `Log_Cosh`<br/>- `Cos_Prox`<br/>- `Poisson_Loss`<br/>- `KLD`                                                                                     |
| `no_of_neurons_per_hidden_layer` | Any `int`values that are at least `1`.<br>If you set it to more than 1 number, e.g. `1, 2`, `2 hidden-layers` will be created.<br> - `1` at first hidden layer<br> - `2` at second layer |
| `optimizer`                      | - `Adadelta`<br/> - `Adagrad`<br/>- `Adam`<br/>- `Nadam`<br/> - `RMSprop`<br/>- `SGD`<br/>- `FRTL`                                                                                       |                                                                                          |

<br><br>

## Data Analysis

---

#### <u>[1] Getting Information from Tensors</u>

* Shape
    * `tensor.shape`
* Rank
    * `tensor.ndim`
* Axis or Dimension
    * `tensor[0]`
* Size
    * `tf.size(tensor)`

#### <u>[2] Convert 2D to 3D Tensor</u>

```python
import tensorflow as tf

rank_2_tensor = tf.constant([[10, 7],
                             [3, 4]])
rank_3_tensor = tf.expand_dims(input=rank_2_tensor,
                               axis=-1)  # <- will set the additional dimension at last
```

#### <u>[3] Changing Tensors Datatype</u>

There are times you will want to downcast tensors datatype. E.g., by default tensors with float values come in float32.
<br>Unless the float value is super large, it is possible to downsize to float16. Modern GPUs have specialized hardware
that handles 16-bits computation.

```python
import tensorflow as tf
from typing import List


# Create a new Tensor with default datatype (float32)
def cast_down(array_tf: List[float], D_Type=tf.float16):
    B = tf.constant(array_tf)
    B_casted = tf.cast(B, dtype=D_Type)
    return B_casted

```

#### <u>[4] Aggregating Tensors</u>

<u>Definition</u>

* Reduce data from multiple values down to smaller values

<u>Purpose</u>

* Improve training efficiency
* Handling large datasets
* Scale Models

<u>Applications</u>

* Parallelism
    * Model
        * Train Single Model that are too large for a device to handle onto multiple devices
            * Each device process a portion of data (mini-batch), and will generate a gradient each.
            * Each device will use either of the following `aggregation` method to combine these generated gradients.
                * `synchronous gradient aggregation`
                * `asynchronous gradient aggregation`
            * Once these gradients are combined, they will be used to update the training model parameters
    * Data
* Distributed Inference
* Dederated Learning
* Large-scale hyperparameter Tuning
* Ensemble Learning
* Real-Time Analytics
* Distributed Reinforcement Learning

<u>Code and Application for `tf.reduce_min()` and `tf.reduce_max()`</u>

```python
import tensorflow as tf
from typing import List


# Normalize Data
def tensor_normalize_data(list_of_tensors: List[int]):
    data = tf.constant(list_of_tensors)
    value_min = tf.reduce_min(data)
    value_max = tf.reduce_max(data)
    normalized_data = (data - value_min) / (value_max - value_min)
    return normalized_data


tensor_normalize_data([1, 2, 3, 4, 5, 6])
```

#### <u>[5] One-Hot Encoding</u>

Use `tf.range(len(cat_cols_raw))`, where `cat_cols_raw` is the list of strings.

```python
import tensorflow as tf
from typing import List

import vic_lim_wx as vic


def convert_cols_to_one_hot_compatible(cat_cols_raw: List[str]):
    logger = vic.Vic_Custom_Logger(config_file="C:/GIT/TF_Exam/tutorials/000_config/config_logger.ini").get_logger()
    color_indices: List[int] = tf.range(len(cat_cols_raw))

    logger.info(f"Converted indices to tensors\n{color_indices}")
    return color_indices
```

<br><br>

## Preprocessing Data

---

#### <u>[1] Data Cleaning Cycle</u>

| Step No | Step Name                   | Step Details                                                                                                             |
|--------:|:----------------------------|--------------------------------------------------------------------------------------------------------------------------|
|       1 | Import data                 | - Get from source file or directly via API call<br/>- Convert into `pd.DataFrame`                                        |
|       2 | Merge Data Sets             | - Optional Step<br/>- If more than 1 data source, will need to merge dataframes                                          |
|       3 | Deduplication               | - Remove duplicated records from dataset                                                                                 |
|       4 | Rebuilding Missing Data     | - <u>Remove</u> Missing Data<br/>- <u>Impute</u> Missing Data                                                            |
|       5 | Standardization             | - Standardize <u>capitalization</u><br>- Standardize <u>date format</u><br>                                              |
|       6 | Normalization               | - <u>Transform</u> data into format suitable for analysis                                                                |
|       7 | Data Downsizing             | - Convert all datatype down to minimum `DType` by using `Downsize_DataFrame()`                                           |
|       8 | Verification and Enrichment | - Check data for accuracy and completeness<br>- Adding additional information                                            |
|       9 | Export Data                 | - Export the data out to the following<br><ul><li>Data Warehouse</li><li>Database</li><li>DataFile (.db, .csv)</li></ul> |

### <u>[2] Feature Selection</u>

<br><br>

## Split Data

---
<br><br>

## Model Training

| Step No | Step Name                   |
|--------:|:----------------------------|
|       1 | Algorithm Selection         |
|       2 | Algorithm Parameters Tuning |
|       3 | Model Fitting               |

---
<br><br>

## Model Evaluation

| Step No | Step Name        |
|--------:|:-----------------|
|       1 | Model Prediction |
|       2 | Model Evaluation |

---
<br><br>

## Hyperparameter Optimization

---
<br><br>

## Cross Validation

---