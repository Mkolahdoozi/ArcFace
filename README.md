# ArcFace
This repo provides TensorFlow 2.X implementation of the famous ArcFace loss for face idnetification. The original paper can be found here:

https://arxiv.org/abs/1801.07698

The official MXNet implemetation can also be found in: 

https://github.com/deepinsight/insightface

However, MXNet is retired: 
https://mxnet.apache.org/versions/1.9.1/


## How to Use
You need to use the implemented ArcFaceLayer() as the last layer in your model. That's it! 
ArcFaceLayer() implements a fully connected layer, as described in the paper, and it adds the arc face loss to the model's total loss using the famous self.add_loss() function in tensorflow. So, if you are planning to use ArcFaceLayer() as the last layer of your model, you don't need to add a fully connected layer at the end of your model.
If you are using Sequential API, using ArcFaceLayer is as easy as:

```python
# here base_model does not have the last fully connected layer. Rather, ArcFaceLayer acts as a fully connected layer.
model = keras.Sequential(
        base_model(),
        ArcFaceLayer()
    ]
```

## How to Train Your Model with ArcFaceLayer()
If you are using model.fit() API to train the model, you don't have to be worry about anything! Just add the ArcFace() to your base model as the above and model.fit() will take care of everything.

If you are writing a training loop from scratch, or customizing training_step(), you have to take care of the added Arc Face loss as the following:

```python
with tf.GradientTape() as tape:

  # Arc Face loss will be available in self.losses attribute, so just add it to other losses you are employing!
  # Here 'loss' represents the other losses you are using.
  loss = tf.math.add(loss, tf.math.add_n(self.losses))

grads = tape.gradient(loss, model.trainable_weights)
optimizer.apply_gradients(zip(grads, model.trainable_weights))
```


For questions and feedback, contact me at 19mk73@queensu.ca
