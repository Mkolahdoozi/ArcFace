import keras
import tensorflow as tf
from math import pi


# ArcFaceLayer acts as a fully connected layer! In addition, it is applying some normalizations that is necessary for
# computing Arc Face loss. Please use this layer as the last layer of your model, the output of which you use
# to do the predictions.
class ArcFaceLayer(keras.layers.Layer):
    def __init__(self, num_classes=3, **kwargs):
        super(ArcFaceLayer, self).__init__(**kwargs)

        # number of the classes for the model
        self.num_classes = num_classes

    def build(self, input_shape):
        self.W = self.add_weight(shape=(input_shape[-1], self.num_classes), trainable=True,
                                 initializer='random_normal', name='weights')

    @tf.function
    def call(self, inputs):
        W = tf.math.l2_normalize(self.W, axis=0)
        inputs = tf.math.l2_normalize(inputs, axis=-1)
        inputs = tf.math.multiply(inputs, self.s) # making the norm od inputs equal to s

        # cos_theta is the output of this layer. Each entry o_ij in the output is equal to s*cos(theta_ij)
        # where s is the scale and theta_ij is the dot product of x_i (ith sample in the input batch) and w_j (jth
        # column of the weigh matrix).
        cos_theta = tf.linalg.matmul(inputs, W)

        return cos_theta



















# This implements the famous Arc Face loss in TF v 2.X. The paper can be found in : https://arxiv.org/abs/1801.07698
# This loss works as expected if you use the implemented ArcFaceLayer as the last layer of your model!
# example:

# keras.Sequential([base_model(), ArcFaceLayer()]) # ArcFaceLayer acts as a dense layer.

# if you are planning to use model.fit() API, simply pass ArcFaceLoss to the model.compile(). Example:

# model.compile(loss=ArcFaceLoss())

# if you are planning to write a training loop from scratch, you have to call an object of a ArcFaceLoss class inside
# your training loop, under the with tf.GradientTape() as tape. Example:

# with tf.GradientTape() as tape:
#   loss = ArcFaceLoss_object(y_true, y_pred)

# In the above code y_true is one-hot encoded labels (it must be a one-hot encoded matrix), y_pred is the output of
# ArcFaceLayer() (last layer of the model) and ArcFaceLoss_object is the object of the ArcFaceLoss class.
class ArcFaceLoss(keras.losses.Loss):
    def __init__(self, margin=0.5, s=64, name='arc_face_loss', **kwargs):
        super(ArcFaceLoss, self).__init__(name, **kwargs)
        self.margin = margin
        self.s = s

    @tf.function
    def call(self, y_true, y_pred):

        cos_theta = tf.math.divide(y_pred, self.s)

        sin_theta = tf.math.sqrt(tf.clip_by_value(tf.math.subtract(1, tf.math.square(cos_theta)), clip_value_min=0,
                                                  clip_value_max=1))

        # cos (theta + margin) = cos(theta)cos(margin)-sin(theta)sin(margin)
        beta = tf.math.multiply(cos_theta, tf.math.cos(self.margin)) - tf.math.multiply(sin_theta,
                                                                                        tf.math.sin(self.margin))

        # to make sure that cos(theta+margin) is monotonically decreasing when theta is in [0, pi] radians
        # you can see the issue here: https://github.com/deepinsight/insightface/issues/108
        threshold = tf.math.cos(pi - self.margin)

        beta_safe = tf.where(cos_theta > threshold,
                             beta,
                             cos_theta - tf.math.multiply(tf.math.sin(pi - self.margin) * self.margin))

        cosine_logits = tf.where(y_true, tf.multiply(beta_safe, self.s), tf.multiply(cos_theta, self.s))

        return tf.nn.softmax_cross_entropy_with_logits(y_true, cosine_logits)




