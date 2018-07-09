# Semantic Segmentation

Project 1 of term 3 of Udacity self-driving car nanodegree

## Table of Contents
1. [Setup](#setup)  
2. [Usage](#usage) 
3. [Files](#files)
4. [Implementation](#implementation)  
    4.1 [Dataset](#dataset)   
    4.2 [Architecture](#architecture)  
    * [Encoder](#encoder)  
    * [1x1 Convolutions](#1x1-convolutions)  
    * [Decoder](#decoder)   

    4.3 [Training](#training)  
    4.4 [Scoring](#scoring)  
    4.5 [Samples of segmented test images](#samples-of-segmented-test-images)

## Setup

Follow instructions [here](https://github.com/udacity/CarND-Semantic-Segmentation-Project) to set up an environment for this project.

## Usage

1. Set up project environemnt as specified above.
2. Run `python main.py` to train a model and segment test images
    
## Files

* main.py - training and scoring
* helper.py - auxiliary functions
* project_tests.py - unit testing functions
* training_log.txt - log file of the training process
* architecture.png - architecture diagram
* scoring_results - folder of segmented test images
* training_samples - folder of the sample training images

## Implementation
The goal of semantic segmentation is to classify each pixel in the image. In this project we used a fully convolutional neural network to label all pixels that belong to a road. This network was trained and tested over [KITTI Road](http://www.cvlibs.net/datasets/kitti/eval_road.php) dataset that can be downloaded from [here](http://www.cvlibs.net/download.php?file=data_road.zip).

[image0]: ./training_samples/um_000000.png "Train image"
[image1]: ./training_samples/um_road_000000.png "Train labels"
[image2]: ./architecture.png "Architecture"
[image3]: ./scoring_results/um_000005.png  "Test image 1"
[image4]: ./scoring_results/umm_000093.png  "Test image 2"
[image5]: ./scoring_results/uu_000003.png  "Test image 3"
[image6]: ./scoring_results/um_000032.png  "Test image 4"

### Dataset
The dataset has 289 training and 290 testing images. All pixels of training images are classified into 3 classes: current road, side road and background. Here is an example of training image
![alt text][image0]
and its labeled pixels
![alt text][image1]
The current road is marked with pink, other road is marked with black and background is marked with red. The road pixels are a union of pink and black pixels. 

### Architecture
Fully convolutional network consists of encoder, 1x1 convolutions, decoder.  

#### Encoder
We used the first 5 convolutional and max pooling layers of VGG16 network. The architecture of these layers can be found [here](https://arxiv.org/pdf/1409.1556.pdf), Table 1, column D. 

#### 1x1 Convolutions
The output of encoder is an input to the layer of 1x1 convolutions. 1x1 convolutions layer is equivalent to fully connected layers of the original VGG16 network. However, unlike fully connected layers, 1x1 convolutions layer allows the pretrained network to segment images of any size. The architecture of 1x1 convolutions layer is:

    4096 convolutions of size 7x7 with stride 1
    4096 RELU activation unit
    Dropout with probability 0.5
    4096 convolutions of size 1x1
    4096 RELU activation units
    Dropout with probability 0.5
    2 convolutions of size 1x1 

#### Decoder
Decoder layer upsamples the output of 1x1 convolutions layer to the size of the original image. We used three upsampling layers:  

    [D1] 2 transposed convolutions of size 4x4 with stride 2 and 'same' padding  
    [D2] 2 transposed convolutions of size 4x4 with stride 2 and 'same' padding  
    [D3] 2 transposed convolutions of size 16x16 with stride 8 and 'same' padding

We also used skip connections to connect the output of the intermediate layers of encoder with the input of the final layers of decoder. In particular, we used the scaled outputs of POOL3 and POOL4 layers of encoder, projected to have depth 2. The complete architecture of decoder layer and its connections to previous layers are shown below. 
![alt text][image2]

### Training 

We used a pre-trained VGG16 network that is available [here](https://s3-us-west-1.amazonaws.com/udacity-selfdrivingcar/vgg.zip).  We trained the network with the following values of hyperparameter:

    learning rate = 0.001
    keep probability = 0.5
    batch size = 8

We also used L2-regularization of the weights of the last convolutional and all transposed convolutional layers. The regularization weight was set to 0.0001.  

We used a data-driven approach to set up the number of epochs. After each epoch we computed IOU over the training set. We stopped training when there was no improvement of IOU over the last 10 epochs. Then we chose the model that generated the highest IOU. To limit the training process we set the maximal number of epochs to 200, but as we describe below, we didn't reach this upper bound.

`training_log.txt` file contains a complete log of our training process, including cross-entropy error and IOU after each epoch. The training process finished after 92 epochs, with the best model being the one obtained after epoch 82. This model has cross-entropy error 0.0254 and IOU 0.9677. We chose this model to segment the test images.

### Scoring 

The image generated by our network has depth 2, with each pixel having two real-valued scores. The first score is for background, the second one is for the road. We used softmax function to convert these scores to probabilities. If a pixel has road probability larger than 0.5 then it is labeled as road, otherwise it is labeled as background.

### Samples of segmented test images

In this section we show several images segmented using our model. The road is marked with green, all other pixels are labelled as background. The complete set of segmented images is in `scoring_results` folder.
![alt text][image3]
![alt text][image4]
![alt text][image5]
![alt text][image6]
