"""
Function: This script is used to show how TensorFlow embedding layer works. TensorFlow now uses Keras to implement deep
          learning layer. Thus this is also an example of Keras embedding layer
Author: Du Fei
Create Time: 2020/7/5 11:21
"""

import numpy as np
import tensorflow as tf


class EmbeddingModel(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, embedding_weights, trainable=False):
        super(EmbeddingModel, self).__init__()
        self.embedding_layer = tf.keras.layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_size,
            embeddings_initializer=tf.keras.initializers.Constant(embedding_weights),
            trainable=trainable
        )

    def call(self, inputs, training=None, mask=None):
        return self.embedding_layer(inputs)


def init_embedding_matrix(vocab_size, embedding_size):
    embedding_weights = np.empty([vocab_size, embedding_size])
    for i in range(vocab_size):
        embedding_weights[i] = i

    return embedding_weights


vocabulary_size = 9
embedding_dims = 4

input_sentence = tf.convert_to_tensor([[0, 1, 2], [4, 2, 6]])
embedding_matrix = init_embedding_matrix(vocabulary_size, embedding_dims)

embedding_model = EmbeddingModel(vocabulary_size, embedding_dims, embedding_matrix, trainable=False)
embedding_out = embedding_model(input_sentence)

init = tf.initialize_all_variables()
with tf.Session() as sess:
    sess.run(init)
    print(sess.run(embedding_out))
