"""
Function: This script is an example of how to use SimpleRNNCell.
          Blog: https://www.datalearner.com/blog/1051594527453718
Author: Du Fei
Create Time: 2020/7/11 12:14
"""

import numpy as np
import tensorflow as tf

input_x = tf.convert_to_tensor(np.asarray([[1, 1, 1], [2, 2, 2]]).astype(np.float32))
previous_state = tf.convert_to_tensor(np.asarray([[[0.1], [0.1]]]).astype(np.float32))

# The units is the only required parameter which represents the dimensions of hidden states.
rnn_cell = tf.keras.layers.SimpleRNNCell(units=1)


# SimpleRNNCell inputs are two variables which the first one is the features and the second one is the previous state.
# The shape of features: [batch_size, feature_size]
# The previous state is either a list of 3D array. Since RNNCell has lots of variants like GRUCell or LSTMCell. The
# cell may contains multiple states. Thus, Keras use a list or 3D array as the input. When the Cell is the
# SimpleRNNCell, the state list only contains one element where the shape is [batch_size, units].
# If the state is a 3D array, then the shape is [1, batch_size, units]

# SimpleRNNCell outputs are also two variables which the first one is the output of cell and the second one is the
# next state. The next state variable is also a list may contains multiple variables which depends on the types of cell.
# Here, in our SimpleRNNCell example, the next_state only contains one variable.
rnn_output, next_state = rnn_cell(input_x, previous_state)

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    rnn_output, next_states = sess.run([rnn_output, next_state])
    print(rnn_output.shape)
    print(next_states[0].shape)
