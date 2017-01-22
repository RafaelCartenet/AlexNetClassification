# AlexNetClassification
Image classification using AlexNet.  
AlexNet is a convolutional neural network that consists of 5 convolutional layers and 3 fully- connected layers. This neural network was first proposed by Alex Krizhevsky, et al. [1], which won the first place at ILSVRC 2012, achieving a top-5 error rate of 15.3%.  
We load a pre-trained AlexNet, read an image, and perform image classification that outputs 5 most probable categories and their probabilities.

AlexNet classifies picture between 1000 categories. For this practice we use a 


# Installation

Pull the content of this repository onto your machine.  

Make sure you have these packages included in your environment :
* tensorflow
* PIL
* numpy

Download the pre-trained Alex Net model :
```
$ wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
```

# Usage
## Predict
We load a pre-trained model of AlexNet so we don't have to train it again. Output is a prediction of the 5 most probable classes.  
In order to predict the categories of a an image test.jpg :  
```
python alexnet_predict.py test.jpg
```
