**Behavioral Cloning** 

Author: Jean-Marc Le Peuvédic

Date: Sunday, September 17th, 2017

Version: 1.0

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

This is a first complete version of the project, which meets the minimum review criteria. Further experimentation with different network architectures are ongoing, and will be included in a final version.

[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/recovery2.png "Recovery Image 1"
[image3]: ./examples/recovery4.png "Recovery Image 2"
[image4]: ./examples/recovery6.png "Recovery Image 3"
[image5]: ./examples/recovery8.png "Recovery Image 4"
[image6]: ./examples/normal.png "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"
[image8]: ./examples/recovery.gif "Recovery animation"
[image9]: ./examples/left_2017_09_13_03_17_55_369.jpg "Left camera"
[image10]: ./examples/center_2017_09_13_03_17_55_369.jpg "Center camera"
[image11]: ./examples/right_2017_09_13_03_17_55_369.jpg "Right camera"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---

###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizing the results
* env.sh sets up the environment for correct execution of the model

model.py and model.h5 are entirely original code. drive.py includes some modifications which match the image preprocessing done for training. 

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
source env.sh
python drive.py model.h5
```
Beware: due to the large amount of data, I decided to find a solution relying on my own hardware rather than relying on EC2 GPU
instances. Having switched brands for AMD years ago due to a more integrated user experience on linux systems, I quickly found out that
tensorflow would not work without CUDA libraries. After upgrading the device drivers, I got a functional configuration using Keras with
the Theano backend, and OpenCL low level operations. I have not yet been able to assess the performance of this setup, even by
comparison. The env.sh script sets up the environment variables for the correct Keras-Theano configuration.

Keras does an excellent job of abstracting out the backend. Indeed, my chosen configuration keeps the tensorflow channel order for
feature maps containing images. Sadly, it is not yet completely possible to just save a model trained on the Theano backend, and load it
in another Keras instance running the tensorflow backend. The correct process is documented here:
https://github.com/fchollet/keras/wiki/Converting-convolution-kernels-from-Theano-to-TensorFlow-and-vice-versa

The Theano backend will run with CUDA on NVidia hardware, but in order to run the model for predictions (no training), the CPU should be
enough.

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for
training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

My model is an adaptation of the NVidia autonomous car architecture. It is identical to the architecture which appears in the course
video. The goal was to reach the required level of performance as quickly as possible, with minimum risk, due to significant changes in
the underlying software and hardware. More precisely, I did not want to get confused by an inability to learn which could have been
caused by a non-functional software/hardware stack. So I got sure that I used a model architecture which would learn with enough
examples.

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 187-212). 
After the convolutional layers, the tensor is flattened and dense layers with 100, 50, 10 and finally 1 perceptrons compute the steering angle.

The model includes RELU layers to introduce nonlinearity (code lines 191, 193, 195, 197, 199), and the data is normalized in the model using a Keras lambda layer (code line 189). 

####2. Attempts to reduce overfitting in the model

The largest number of times certain image subsets have been used for training is 60. Each simulator run generated its own driving log
file, and the files had to be concatenated (cat driving_log*.csv > driving_log.csv) before the training. Due to errors in handling those
log files, some subsets were included more than once in the training data. As a consequence, the validation set almost certainly
contained images in part identical to some training images. It was therefore difficult to judge overfitting, but the 14 hour long
training could not be easily repeated, so a more economic solution was applied.

The duplicates were eliminated from the driving log, and a couples of fine tuning epochs were added (about 15 minutes). This solution
validated to capability to start and stop training at will using Keras, and therefore the possibility to save the weights after each
epoch if needed. This possibility breaks the history_object returned by model.fit_generator, which I use to visualize training and
validation loss. Further work is needed to merge the successive history_object returned by successive calls to model.fit.

The fine tuning did not reveal a need to reduce overfitting : both training and validation loss were low, and got lower. Since I did not
use any regularization technique to avoid overfitting, I believe that including 1364 images (17%) from circuit 2 was important to help
the model to generalize well and avoid overfitting. 

The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 213).
The batch size was mostly set for overall performance, since the computer had to handle simultaneously a large amount of I/O, CPU
basedpre-processing and GPU-based training. The final compromise is 192.

It is important to note that images are flipped in memory by the generator, so each batch is balanced in terms of curves to the left or
to the right. 

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left
and right sides of the road with various types of pavement and side lines, and I also used the left and right camera images. 

The data set contains three times 8017 images in total and the associated steering angle. More than one thousand image have been
manually deleted because they exhibited poor driving behaviour. 

* The first subset includes three laps in either direction (the half-turn
manoeuver in one of the earth pockets of the track had to be deleted). Every time I drove off center and too close to a side of the
pavement, a few images of the divergence instants have been manually deleted and the recovery was left;
* The second subset includes 911 track locations (and a number of images equal to three times that amount) : mostly recoveries in specific areas of the track, where the car tended to drive over the edges
of the pavement, or to drive away from the track;
* The third subset includes 1364 track locations on track 2, and it is essential to avoid overfitting;
* Finally the last subset contains 574 track locations, mostly recoveries in areas the car had difficulties to cross.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
