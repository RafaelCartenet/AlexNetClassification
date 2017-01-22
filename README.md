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

Install the following packages inside your environment.  
Pip installation :  
```
$ pip install h5py
$ pip install tflearn
```
Anaconda installation :
```
$ conda install h5py
$ conda install tflearn
```

Download the pre-trained Alex Net model :
```
$ wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
```

Download Caltech 101 dataset, which can be done as follows at a Linux command prompt.
```
$ wget --no-check-certificate https://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz 
$ tar xvzf 101_ObjectCategories.tar.gz
```

Generate the .h5 files from the dataset, has to be done only once.
```
python HDF5_generation.py
```

# Usage

## Training
Once you have completed installation step you can train the model. In order to train the model simply execute :
```
python alexnet_training.py
```

## Predict
Once your model is trained you can use it in order to predict the class of a given jpg picture. Output is a prediction of the 5 most probable classes.  
In order to predict the categories of a an image test.jpg :  
```
python alexnet_predict.py test.jpg
```
