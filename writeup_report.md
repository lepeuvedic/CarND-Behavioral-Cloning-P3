**Behavioral Cloning** 

Author: Jean-Marc Le Peuvédic

Date: Sunday, September 17th to 20th, 2017

Version: 1.0

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

This is a first complete version of the project, which meets the minimum review criteria. Further experimentation with different network
architectures are ongoing, and will be included in a final version.

[//]: # (Image References)

[image1]: ./examples/model.png "Model Visualization"
[image2]: ./examples/recovery2.png "Recovery Image 1"
[image3]: ./examples/recovery4.png "Recovery Image 2"
[image4]: ./examples/recovery6.png "Recovery Image 3"
[image5]: ./examples/recovery8.png "Recovery Image 4"
[image6]: ./examples/normal.png "Normal Image"
[image7]: ./examples/flipped.png "Flipped Image"
[image8]: ./examples/recovery.gif "Recovery animation"
[image9]: ./examples/left_2017_09_13_03_17_55_369.jpg "Left camera from the same location"
[image10]: ./examples/center_2017_09_13_03_17_55_369.jpg "Center camera"
[image11]: ./examples/right_2017_09_13_03_17_55_369.jpg "Right camera from the same location"
[image12]: ./examples/recovery_angle.png "Recovery angle"
[image13]: ./examples/training_convergence.png "Training convergence show no sign of overfitting"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each
point in my implementation.  

---

### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* `model.py` containing the script to create and train the model
* `drive.py` for driving the car in autonomous mode
* `model.h5` containing a trained convolution neural network 
* `writeup_report.md` summarizing the results
* `env.sh` sets up the environment to use Keras with Theano (required for the correct execution of the model)

model.py and model.h5 are entirely original code. drive.py includes some modifications which match the image preprocessing done for
training. 

#### 2. Submission includes functional code
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

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for
training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

My model is an adaptation of the NVidia autonomous car architecture. It is identical to the architecture which appears in the course
video. The goal was to reach the required level of performance as quickly as possible, with minimum risk, due to significant changes in
the underlying software and hardware. More precisely, I did not want to get confused by an inability to learn which could have been
caused by a non-functional software/hardware stack. So I got sure that I used a model architecture which would learn with enough
examples.

My model consists of a convolution neural network with 5x5 and 3x3 filter sizes and depths between 24 and 64 (model.py lines 187-212). 
After the convolutional layers, the tensor is flattened and dense layers with 100, 50, 10 and finally 1 perceptrons compute the steering
angle.

The model includes RELU layers to introduce nonlinearity (code lines 191, 193, 195, 197, 199), and the data is normalized in the model
using a Keras lambda layer (code line 189). 

#### 2. Attempts to reduce overfitting in the model

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

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 213).
The batch size was mostly set for overall performance, since the computer had to handle simultaneously a large amount of I/O, CPU
basedpre-processing and GPU-based training. The final compromise is 192.

It is important to note that images are flipped in memory by the generator, so each batch is balanced in terms of curves to the left or
to the right. 

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left
and right sides of the road with various types of pavement and side lines, and I also used the left and right camera images. 

The data set contains three times 8017 images in total and the associated steering angle. More than one thousand image have been
manually deleted because they exhibited poor driving behaviour. 

* The first subset includes three laps in either direction (the half-turn
manoeuver in one of the earth pockets of the track had to be deleted). Every time I drove off center and too close to a side of the
pavement, a few images of the divergence instants have been manually deleted and the recovery was left;
* The second subset includes 911 track locations (and a number of images equal to three times that amount) : mostly recoveries in
specific areas of the track, where the car tended to drive over the edges
of the pavement, or to drive away from the track;
* The third subset includes 1364 track locations on track 2, and it is essential to avoid overfitting;
* Finally the last subset contains 574 track locations, mostly recoveries in areas the car had difficulties to cross.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to copy the model of the instructional video. I know that it's a cheap
strategy, but the result was guaranteed from multiple sources. Further tests have been made, espacially with an inception model which
mixes various smaller, preprocessed versions of the images, but the results are not yet available.

My first step was to use a convolution neural network model similar to the NVidia architecture. I thought this model might be
appropriate because it is the right tool for a machine vision application. I made a few test with HSV colorspace conversion, which
provides a grayscale channel and a hue channel where, I hoped, the gray pavement would be easily recognizable. The approach was not more
successful than using RGB.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set (model.py
line 134). The first attempts at training were tests with 5 to 9 epochs and somewhat larger dataset. Overfitting was clearly not present
at that stage.  

In the final steps I was training with a significant proportion (17%) of track 2 images in both sets, and the model was not overfitting.
I imagined various strategies should overfitting occur:

* Adding Dropout layers after the convolutional layers and after the first two Dense layers;
* Adding more track 2 images;
* Using the Keras image augmentation function to rotate the pictures slightly.

The addition of Dropout layers did not change the behaviour, but did not seem to provide any advantage, so I eliminated the Dropout
layers from the final model.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle
fell off the track. To improve the driving behavior in these cases, I deleted the corresponding examples of driving off-track from the
samples, and added recovery trajectories from the side of the track recorded in the same sensitive areas.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road. Two complete laps are
recorded in the file `video.mpg`.

#### 2. Final Model Architecture

The final model architecture (model.py lines 187-212) consisted of a convolution neural network with the following layers and layer
sizes :

* Image normalization;
* Convolution layer, valid padding, 5x5 kernel, stride 2, 24 layers;
* Convolution layer, valid padding, 5x5 kernel, stride 2, 36 layers;
* Convolution layer, valid padding, 5x5 kernel, stride 2, 48 layers;
* Convolution layer, valid padding, 3x3 kernel, stride 1, 64 layers;
* Convolution layer, valid padding, 3x3 kernel, stride 1, 64 layers, at which point the map is 1 pixel high;
* Flatten; 
* Dense layer 100 neurons;
* Dense layer 50 neurons;
* Dense layer 10 neurons;
* Output 1 neuron, which controls steering;

Here is a visualization of the architecture obtained using Keras visualisation utility:

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded three laps in each driving direction on track one using center lane driving. Here is
an example image of center lane driving:

![Exemple image recorded from the center camera][image6]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to
recover from excursions to the sides of the track. These images show what a recovery looks like starting from the edge of the pavement :

![Still image from recovery sequence : start][image2]
![Still image from recovery sequence : next][image3]
![Still image from recovery sequence : next][image4]
![Still image from recovery sequence : recovered to center of the track][image5]

The following animated GIF should give a better understanding of the recovery trajectory:

![Animated GIF of a typical recovery sequence][image8]
![evolution of steering angle during recovery trajectory][image12]

The image on the right shows the evolution of the steering angle during a recovery trajectory: it starts with a relatively high value
and is smoothly reduced to the correct value for the long term curve of the track.

Then I added images recorded on track two, driving in the middle line, in order to get very different data points which would help to
fence off overfitting.

To augment the data sat, I also flipped images and angles thinking that this would help balance right over steering and left over
steering tendencies. For example, here is an image that has then been flipped:

![Normal image][image6]
![Flipped image][image7]

Because I had a relatively limited amount of images, I decided to use the left and right cameras. The side cameras provide an off-center
view of the track. All three cameras are perfectly aligned. The method published by NVidia to use that data, is to associated to the
side images a steering angle value derived from the recorded steering angle, as if it was an image from the center camera. In other
terms, the recorded steering angle corresponds to the center camera, and derived steering angles must be computed for the side cameras.

All the dimensions in the landscape can be normalized by the scale of the car, and the most natural measure of that is the distance
between the rear axle and the plane of the cameras. The Z direction is forward, in the general direction of the camera view port depth.
The X direction is to the right side of the car, and the Y direction is vertical, upwards. 

I examined the unity source code of the simulator to get the exact geometric dimensions:

* The distance e between the rear axle and the cameras is 2.61 meters;
* The distance \Delta x between the car centerline and the right camera is 0.825 meters;
* The distance \Delta x between the car centerline and the left camera is 0.806 meters.

If the view from the center camera is the middle of the track, the right camera shows the right side of the track. Therefore a negative
offset to the steering angle is needed to get that view closer to the center of the track. 

Due to the turn radius becoming infinite, two different calculation methods are needed to derive a steering angle offset when the
recorded steering angle is small (close to zero) or large. An important parameter is the distance ahead of the car at which the recovery
should be achieved. It is obvious that this distance will very with speed, and is better expressed as the product of some speed V with a
time delay t. Because I was driving at the maximum speed, 30 mph during the most of the training sessions on the simulator, I finally
used a fixed offset for each side camera. I thought that two seconds to recover seemed appropriate, and therefore the distance covered
in two seconds at 30 mph is the standard recovery distance. In distances normalized by e, the formula is - 2 Xe² / V²t² .

V/e X=x/e are the normalized speed and camera offset.

After the collection process, I had 8017 data points, with left, center and right images for each. Taking into account horizontally
flipped images made by the generator, training uses 48102 images and the associated steering angles. Preprocessing is limited to an affine transform of pixel R,G,B values, which projects the 0..255 range onto the -0.5..0.5 range. The normalization is performed by the lambda layer, which is the first in the network.

![Center camera image][image10]
![Left camera image at the same location][image9]
![right camera image at the same location][image11]

I finally randomly shuffled the data set and put 20% of the data into a validation set, leaving 35534 data points for training. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal
number of epochs was not clearly determined, but rather high, in the order of 30 as evidenced by a plot of traininng and validation
losses reproduced here. I used an adam optimizer so that manually training the learning rate wasn't necessary.

![Convergence of training and validation loss][image13]

