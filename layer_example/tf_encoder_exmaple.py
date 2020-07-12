"""
Function: This script is an example of how to use encoder
Author: Du Fei
Create Time: 2020/7/11 10:55
"""

import numpy as np
import tensorflow as tf

tf.get_logger().setLevel('ERROR')


def init_rnn_input(batch_size, time_steps, time_feature_size):
    """
    This method will generate 3D numpy array where represents [batch_size, time_steps, feature_size]
    :param batch_size:
    :param time_steps:
    :param time_feature_size:
    :return:
    """
    rnn_input = np.empty([batch_size, time_steps, time_feature_size])
    for i in range(batch_size):
        for j in range(time_steps):
            rnn_input[i][j] = j

    return rnn_input


_sample_size = 6
_time_steps = 5
_time_feature_size = 3

rnn_input_example = tf.convert_to_tensor(
    init_rnn_input(_sample_size, _time_steps, _time_feature_size),
    dtype=tf.float32
)

# units is the only required parameter which represents the dimension of hidden states between rnn cells
encoder_model = tf.keras.layers.SimpleRNN(units=3, return_sequences=True, return_state=True)

# RNN Input is a 3D array: [batch_size, time_steps, feature_size]
# RNN output is a list which may contains one or two variables.
# if we set return_sequences=True only, then there is only one return value which represents the states of all time
# steps (or cell). It means we will get a variable where the shape is [batch_size, time_steps, units]
# if we set return_state=True only, then we will get two identical variables where both of them represents the states
# of the last time step (or cell). The shape is [batch_size, units].
# if we set return_sequences=True and return_state=True simultaneously, then we will get two variables which the first
# one represents the states of all time steps (or cell). The shape is [batch_size, time_steps, units].
# And the second variable represents the state of last time steps (or cell). The shape is [batch_size, units]
rnn_states, rnn_last_states = encoder_model(rnn_input_example)

init = tf.compat.v1.global_variables_initializer()
with tf.compat.v1.Session() as sess:
    sess.run(init)
    rnn_output, rnn_states = sess.run([rnn_states, rnn_last_states])
    print(rnn_output.shape)
    print(rnn_states.shape)
