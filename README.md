# Gray-Matter-Lane-Detection

In this project, we have use two approach but the second approach doesnot work well with the Indian Roads.

# First Approach 

In this we use a deep learning-based approach to improve upon lane detection. 
We uses a fully convolutional neural network to output an image of a predicted lane.

<p align="center">
  Lane detection using Deep Learning<br>
  <img width="600" height="400" src="https://github.com/Giscle/Gray-Matter-Lane-Detection/blob/master/output.gif">
</p>

# Software Requirements:

If using Linux or Mac, you can use this conda environment file. In the command line, use conda env create -f lane_environment.yml and then source activate lane_environment to use the environment. This is a slightly modified environment from that used in Term of the Udacity SDCND. Please see the note on moviepy below.

Alternatively, you can use the following:

    Python 3.5 or higher. Most of the included code also works in Python 2.7, although the pickle files used for the training images and labels are encoded for use in Python 3 and would need to be loaded and restructured to work in Python 2.7. I would suggest downloading with Anaconda as you will get some of the below
    TensorFlow - Suggested to download TensorFlow GPU for best performance
    Keras
    OpenCV (known as cv2 once downloaded) - opencv-python if installing from command line
    h5py
    imageio
    matplotlib
    moviepy
    numpy+mkl
    scikit-learn
    scipy


# KEY FILES:

fully_conv_NN.py -This is the fully convolutional neural network to train using that data.

full_CNN_model.h5 - These are the final outputs from the above CNN. Note that if you train the 
file above the originals here will be overwritten! These get fed into the below.

draw_detected_lanes.py - Using the trained model and an input video, this predicts the lane, averages across 5 frames, and returns the original video with predicted lane lines drawn onto it. 

# Second Apporch 

This approach is based on edge detection using opencv followed by masking the image
with region of interest and then applying hough transformation to get the lane lines.
Steps to be followed :-

1.Apply some filtering/blurring to get sharp edges & rid of noises in the roads ( we
can experiment and choose any one filtering from bilateral filter, gabor filter,
trilateral filter )

2.Conversion of image to grayscale ( This is help in improving our processing
time).

3.Apply canny detector, to detect the edges.

4.Apply a ROI mask ( to eliminate the non-required elements from the scene).

5.Apply hough transform (this step also requires a vigorous experimentation with
tweaking of parameters).

6.Detect the lane lines.

7.Mask the lines to the original image.

The above seven steps will mark the end of our Approach 2.

Reference:
https://github.com/naokishibuya/car-finding-lane-lines/
https://github.com/mvirgo/MLND-Capstone
