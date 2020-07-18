# deep_learning_with_python

这个项目主要是用来展示如何使用python中一些与深度学习或者是机器学习有关的模型、层、工具方法等。

主要包括TensorFlow中一些层的使用以及模型的使用

### 依赖（本机环境验证的依赖版本）:

TensorFlow: 1.15.3

Numpy: 1.16.2

---

目前，本项目包括如下内容


### Layer的使用

*layer_example/tf_embedding_example.py*

这个是展示embedding层的使用方法

*layer_example/tf_rnn_cell_example.py*

这个是展示RNN中单独的Cell的输入输出

*layer_example/tf_encoder_example.py*

这个是展示Seq2Seq中Encoder模型如何使用，也就是一个正常的RNN使用方法

### TensorFlow一些工具

*tf_example/tf_keras_progress_bar_example.py*

这个脚本是展示如何使用TensorFlow中的进度条，即tf.keras.util.Progbar的使用

### 数据处理的一些实例

*data_example/normalize_data.py*

正规化是一种非常有用的方法，它经常被用在算法的数据预处理中，这个脚本提供了两个方法，第一个是将pandas.DataFrame变成正规化的数据，并返回均值和方差，第二个方法是根据正规化后的数据、均值和方差来恢复原始数据。在预测任务中，我们经常先要正规化输入数据，并将原始结果恢复。


-----

# deep_learning_with_python

This project will show examples of deep learning model with python.

We will write examples for different deep learning layers or models.

### Dependencies:

TensorFlow: 1.15.3

Numpy: 1.16.2

---

Currently, this project contains examples as follows:


### Layer example

*layer_example/tf_embedding_example.py*

This script is an example of embedding layer using TensorFlow.

*layer_example/tf_rnn_cell_example.py*

This script is an example of SimpleRNNCell layer using TensorFlow.

*layer_example/tf_encoder_example.py*

This script is an example of SimpleRNN layer using TensorFlow (Keras).

### Some TensorFlow utils method

*tf_example/tf_keras_progress_bar_example.py*

The example of how to use tf.keras.util.Progbar

### Data process examples

*data_example/normalize_data.py*

Normalizing data is very useful in deep learning or other machine learning algorithms. This script provides two method which the first one is to normalize pandas.DataFrame, return normalized data, mean values and std values.The second one is to recover raw data by normalized data, mean values and std values.
