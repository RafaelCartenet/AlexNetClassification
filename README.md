# AlexNetClassification
Image classification using AlexNet.  
AlexNet is a convolutional neural network that consists of 5 convolutional layers and 3 fully- connected layers. This neural network was first proposed by Alex Krizhevsky, et al. [1], which won the first place at ILSVRC 2012, achieving a top-5 error rate of 15.3%.  
We load a pre-trained AlexNet, read an image, and perform image classification that outputs 5 most probable categories and their probabilities.



# Usage


1. Install the following packages inside your environment
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

Make sure you also have these packages included in your environment :
* PIL

2. download the pre-trained Alex Net model :
```
$ wget http://www.cs.toronto.edu/~guerzhoy/tf_alexnet/bvlc_alexnet.npy
```

3. Download Caltech 101 dataset, which can be done as follows at a Linux command prompt.
```
$ wget --no-check-certificate https://www.vision.caltech.edu/Image_Datasets/Caltech101/101_ObjectCategories.tar.gz 
$ tar xvzf 101_ObjectCategories.tar.gz

```
3. Generate the .h5 files from the dataset, has to be done only once.
```
python project2_HDF5_generation.py
```

