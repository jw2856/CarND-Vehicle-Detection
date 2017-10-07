## **Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # (Image References)
[hog1]: ./output_images/car_notcar_visualization_hsv_hogchannel1.png
[hog2]: ./output_images/car_notcar_visualization_rgb1_hogchannel0.png
[hog3]: ./output_images/car_notcar_visualization_rgb2_hogchannel1.png
[hog4]: ./output_images/car_notcar_visualization_YCrCb_channel0.png
[hog5]: ./output_images/car_notcar_visualization_YCrCb_channel1.png
[hog6]: ./output_images/car_notcar_visualization_YCrCb_channel2.png

[spatial1]: ./output_images/bin_spatial_1.png
[spatial2]: ./output_images/bin_spatial_2.png

[colorhist1]: ./output_images/color_hist_HLS.png
[colorhist2]: ./output_images/color_hist_HSV.png
[colorhist3]: ./output_images/color_hist_LUV.png
[colorhist4]: ./output_images/color_hist_RGB.png
[colorhist5]: ./output_images/color_hist_YCrCb.png
[colorhist6]: ./output_images/color_hist_YUV.png

[video1]: ./project_video.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Writeup / README

You're reading it!

### Feature Extraction

#### Pipeline

In the `detect.py` file, I provide a function called `extract_features` that takes an array of images and outputs a features array consisting of features extracted from that image given the parameters passed to the function. Internally, this function passes each image to a `single_img_features` function that calculates the features for each singular image.

In `single_img_features`, the features for each image are extracted, and can extract up to three different types of features:

- HOG features
- Spatial binning
- Color histogram

#### HOG Features

HOG stands for Histogram of Oriented Gradients, and is a porcine way of saying that it lets us extract a directional gradient of each pixel. The algorithm calculates the gradient of each pixel and its surrounding pixels, and then returns a histogram that corresponds to the sum of the gradient values in each direction. This results in an image that very roughly detects edges and shapes.

Here are some examples of HOG features extracted for randomly sampled cars and not cars taken from the dataset, using different color spaces and channels:

**HSV Channel 1**

![hog1][hog1]

**RGB Channel 0**

![hog2][hog2]

**RGB Channel 1**

![hog3][hog3]

**YCrCb Channel 0**

![hog4][hog4]

**YCrCb Channel 1**

![hog5][hog5]

**YCrCb Channel 2**

![hog6][hog6]

#### Spatial binning

Spatial binning downsamples an image to a lower resolution, which speeds up processing while (hopefully) preserving the useful information in an image. Here are two examples of spatially binned cars and not cars:

![spatial1][spatial1]

![spatial2][spatial2]

#### Color histogram

Taking a color histogram of an image allows us to extract useful features of an image from the color information of the image. The color information can be encoded in many different formats. Most images use RGB, and indeed when we import images the default encoding format is RGB (or BGR, if using CV2). Unfortunately, RGB encoding doesn't always provide us with the most useful distinctions when extracting features, and thus we convert images into a variety of other color spaces. Our program gives us the option to convert into the HSV, LUV, HLS, YUV, and YCrCb color spaces.

Here are some example histograms of a sample image in various color spaces:

**HLS**

![colorhist1][colorhist1]

**HSV**

![colorhist2][colorhist2]

**LUV**

![colorhist3][colorhist3]

**RGB**

![colorhist4][colorhist4]

**YCrCb**

![colorhist5][colorhist5]

**YUV**

![colorhist6][colorhist6]









The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

I then explored different color spaces and different `skimage.hog()` parameters (`orientations`, `pixels_per_cell`, and `cells_per_block`).  I grabbed random images from each of the two classes and displayed them to get a feel for what the `skimage.hog()` output looks like.

Here is an example using the `YCrCb` color space and HOG parameters of `orientations=8`, `pixels_per_cell=(8, 8)` and `cells_per_block=(2, 2)`:


![alt text][image2]

#### Parameters

I tried various combinations of parameters and...

####3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

###Sliding Window Search

####1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

####2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

####1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


####2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

###Discussion

####1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

