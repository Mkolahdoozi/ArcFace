# ArcFace
This repo provides TensorFlow 2.X implementation of the famous ArcFace loss for face identification. The original paper can be found here:

https://arxiv.org/abs/1801.07698

The official MXNet implementation can also be found in: 

https://github.com/deepinsight/insightface

However, MXNet is retired: 
https://mxnet.apache.org/versions/1.9.1/

Also, there are other implementations for tensorflow as well; however, they are not a plug-n-play layer:
https://github.com/yinguobing/arcface


## How to Use
You need to add the implemented ArcFaceLayer() as the last layer in your model and add the implemented ArcFaceLoss in the
model.compile(). That's it! 

ArcFaceLayer() implements a fully-connected like layer, as described in the paper, along with some normalizations which are essential to calculate ArcFaceLoss.

ArcFaceLoss provides you with the arc face loss, implemented in accordance with the original paper.

If you are planning to use Sequential API, here is an example on how you can use the ArcFaceLayer and ArcFaceLoss:

```python
# here base_model is the base model you are using. We add ArcFaceLayer as the last layer.
from ArcFace import ArcFaceLayer, ArcFaceLoss

model = keras.Sequential([
        base_model(),
        ArcFaceLayer(num_classes=3)
    ])

model.compile(loss=ArcFaceLoss())
model.fit(x, y, epochs=100)
```

If you are implementing a training loop from scratch, or customizing training_step(), you have to 
take care of the ArcFaceLayer and ArcFaceLoss manually. Here is an example:

```python
from ArcFace import ArcFaceLayer, ArcFaceLoss

inputs = keras.Input(shape=(input_shape))
x = base_model(input)
arcface_layer = ArcFaceLayer(num_classes=num_classes)
x = arcface_layer(x)
model = keras.Model(inputs=inputs, outputs=x)

arcface_loss = ArcFaceLoss()

for epoch in range(epochs):
    for x_batch, y_batch in train_data:
        
        with tf.GradientTape() as tape:
            
            logits = model(x_batch, training=True)
            loss = arcface_loss(y_batch, logits)
        
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
```


For questions and feedback, contact me at 19mk73@queensu.ca

Also, see our amazing group at: https://www.aiimlab.com/