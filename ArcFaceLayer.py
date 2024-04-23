import keras
import tensorflow as tf
from math import pi


class ArcFaceLayer(keras.layers.Layer):
    def __init__(self, margin=0.5, s=64, num_classes=3, **kwargs):
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

        cos_theta = tf.math.divide(tf.linalg.matmul(inputs, W), self.s)
        sin_theta = tf.math.sqrt(tf.clip_by_value(tf.math.subtract(1, tf.math.square(cos_theta)), clip_value_min=0,
                                                  clip_value_max=1))

        # cos (theta + margin) = cos(theta)cos(margin)-sin(theta)sin(margin)

        beta = tf.math.multiply(cos_theta, tf.math.cos(self.margin)) - tf.math.multiply(sin_theta,
                                                                                        tf.math.sin(self.margin))

        # to make sure that cos(theta+margin) is monotonically deacreading when theta is in [0, pi] radians
        # you can see the issue here: https://github.com/deepinsight/insightface/issues/108

        threshold = tf.math.cos(pi - self.margin)

        beta_safe = tf.where(cos_theta > threshold,
                             beta,
                             cos_theta - tf.math.multiply(tf.math.sin(pi - self.margin) * self.margin))


