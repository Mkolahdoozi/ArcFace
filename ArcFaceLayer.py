import keras
import tensorflow as tf


class ArcFaceLayer(keras.layers.Layer):
    def __init__(self, margin=0.5, s=64, num_classes=2, **kwargs):
        super(ArcFaceLayer, self).__init__(**kwargs)

        # margin parameter controls the extent of angular penalty. The default value is 0.5 in the paper
        # s parameter (scale) is a parameter that controls the norm of the input. The default is 64 in the paper.
        # num_classes is the number of the logit outputs (classes)
        self.margin = margin
        self.s = s
        self.num_classes = num_classes

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.num_classes), trainable=True,
                                 initializer='random_normal', name='weights')

    @tf.function
    def call(self, inputs):
        W = tf.math.l2_normalize(self.W, axis=0)
        inputs = tf.math.l2_normalize(inputs, axis=-1)
        inputs = tf.math.multiply(inputs, self.s)

        cos = tf.linalg.matmul(W, inputs)










