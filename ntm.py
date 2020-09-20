from tensorflow.keras import models, layers, losses
import tensorflow as tf
import tensorflow.keras as keras

class NTMLayer(layers.Layer):


    def __init__(self, input_shape, Beta, memory_length, memory_size, uints = 625):
        super(NTMLayer, self).__init__()
        self.uints = uints
        self.Beta = Beta
        self.erasing_error = tf.Variable([0], dtype='float32', trainable = True)
        self.writing_error = tf.Variable([0], dtype='float32', trainable = True)
        w_init = tf.random_normal_initializer()
        self.memory_weight = tf.Variable(dtype='float32', initial_value=w_init(shape=[memory_size, memory_length]),
                                         trainable = True)

    @staticmethod
    def cosine_similarity(k, m):
        k_mag = tf.sqrt(tf.reduce_sum(tf.square(k), axis=-1))
        m_mag = tf.sqrt(tf.reduce_sum(tf.square(m), axis=-1))
        mag_prod = tf.multiply(k_mag, m_mag)
        dot = tf.squeeze(tf.keras.layers.dot([k, m], axes=(-1, -1)), axis=1)
        return tf.divide(dot, mag_prod)
    def call(self, inputs):

        return tf.matmul(inputs, self.w)

l = NTMLayer(input_shape= [2], memory_length= 20, memory_size= 30, Beta=0.3, uints = 3)
x = tf.ones([2])
print(x)
print(l.w)
print(l([x]))