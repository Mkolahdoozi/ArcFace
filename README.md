# ArcFace
This repo provides TensorFlow 2.X implementation of the famous ArcFace loss for face idnetification. The original paper can be found here:

https://arxiv.org/abs/1801.07698

The official MXNet implemetation can also be found in: 

https://github.com/deepinsight/insightface

However, MXNet is retired: https://mxnet.apache.org/versions/1.9.1/


## How to Use
You need to use the implemented ArcFaceLayer() as the last layer in your model. That's it! 
ArcFaceLayer() implements a fully connected layer, as described in the paper, and it adds the arc face loss to the model using the famous self.add_loss() in tensorflow. If you are using model.fit() to train the model, you don't have to be worry about anything! Just add the ArcFace to your base model as the following:

In the sequential API:

```python
model = keras.Sequential(
        base_model(),
        ArcFaceLayer()
    ]
```

If you are writing the training loop from scratch, or editing training_step(), you have to take care of the added Arc Face loss as the following:

```python
  with tf.GradientTape() as tape:
  
      # Arc Face loss will be available in self.losses attribute, so just add it to other losses you are employing!
      # Here 'loss' is the other loss you are using.
      loss = tf.math.add(loss, tf.math.add_n(self.losses))
```


For questions and feedback, contact me at 19mk73@queensu.ca
