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

[testmodel_rgb]: ./output_images/test_images_result_rgb_128_window_50_overlap.png
[testmodel_hls]: ./output_images/test_images_result_hls_128_window_50_overlap.png
[testmodel_hsv]: ./output_images/test_images_result_hsv_128_window_50_overlap.png
[testmodel_luv]: ./output_images/test_images_result_luv_128_window_50_overlap.png
[testmodel_ycrcb]: ./output_images/test_images_result_YCrCb_128_window_50_overlap.png
[testmodel_yuv]: ./output_images/test_images_result_yuv_128_window_50_overlap.png

[model_test_images]: ./output_images/test_images_result_YCrCb_3windows_75_overlap_2threshold.png

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

Here are some examples of HOG features extracted for randomly sampled cars and not cars taken from the dataset, using orient=9, pix_per_cell=8, and cell_per_block=2, in different color spaces and channels:

*HSV Channel 1*

![hog1][hog1]

*RGB Channel 0*

![hog2][hog2]

*RGB Channel 1*

![hog3][hog3]

*YCrCb Channel 0*

![hog4][hog4]

*YCrCb Channel 1*

![hog5][hog5]

*YCrCb Channel 2*

![hog6][hog6]

#### Spatial binning

Spatial binning downsamples an image to a lower resolution, which speeds up processing while (hopefully) preserving the useful information in an image. Here are two examples of spatially binned cars and not cars:

![spatial1][spatial1]

![spatial2][spatial2]

#### Color histogram

Taking a color histogram of an image allows us to extract useful features of an image from the color information of the image. The color information can be encoded in many different formats. Most images use RGB, and indeed when we import images the default encoding format is RGB (or BGR, if using CV2). Unfortunately, RGB encoding doesn't always provide us with the most useful distinctions when extracting features, and thus we convert images into a variety of other color spaces. Our program gives us the option to convert into the HSV, LUV, HLS, YUV, and YCrCb color spaces.

Here are some example histograms of a sample image in various color spaces:

*HLS*

![colorhist1][colorhist1]

*HSV*

![colorhist2][colorhist2]

*LUV*

![colorhist3][colorhist3]

*RGB*

![colorhist4][colorhist4]

*YCrCb*

![colorhist5][colorhist5]

*YUV*

![colorhist6][colorhist6]

#### Parameters

To select parameters, I trained an SVM on a subset of the dataset using each of the colorspaces three times each using identical orientation, pixels per cell, cells per block, and hog channel settings. I then took the average of the accuracies for each on the models on the validation set. Here were the results:

RGB: 93.667%
HSV: 94.167%
LUV: 93.667%
HLS: 94.833%
YUV: 93.833%
YCrCb: 94.33%

From these results, I took the HLS, HSV, and YCrCb, and varied the orientation, pixels per cell, cells per block, and hog channel parameters to see the impact on the validation accuracy. The best result I achieved at this point was with an HLS color space, using `hog_channel=ALL`, `orient=9`, `pix_per_cell=8` and `cell_per_block=2`.

However, to be more confident about my choice of parameters, I wanted to have a visualization of how each classifier performed on the test images. So I trained various models using different color spaces with the other parameters held constant and ran the test images through them to see how them performed.

*HLS, windows=(128, 128), 50% overlap*

![testmodel_hls][testmodel_hls]

*HSV, windows=(128, 128), 50% overlap*

![testmodel_hsv][testmodel_hsv]

*LUV, windows=(128, 128), 50% overlap*

![testmodel_luv][testmodel_luv]

*RGB, windows=(128, 128), 50% overlap*

![testmodel_rgb][testmodel_rgb]

*YCrCb, windows=(128, 128), 50% overlap*

![testmodel_ycrcb][testmodel_ycrcb]

*YUV, windows=(128, 128), 50% overlap*

![testmodel_yuv][testmodel_yuv]

Based on these results, YCrCb and HLS seems to do pretty well, while the other choices gave quite a few false positives. I decided to use YCrCb since it seemed to do a better job at detecting the white car, while HLS seemed to miss it.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

In `main.py`, I read the `cars` and `notcars` from the provided data. I calculate a scaler to scale the features to the same range using `StandardScaler`, use `train_test_split` to create a random test/validation split, and then train a linear SVM. I save the model and the scaler to disk so that I don't have to train the model each time to test the model with different parameters.

The different models for different color scales are in the `models` folder. For the final video, I used the `model-YCrCb.pkl` file in the main project directory.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.

I first ran the model over the test images in the test directory and generated a heatmap of the results. Having decided on using YCrCb, I experimented with different window sizes and overlaps. I found that 128 pixel windows were a bit too large for some of the cars further in the distance, and 96 and 64 pixel windows were able to detect cars in some places where the 128 pixel images weren't. In addition, by using a 75% overlap, I was able to get a stronger heatmap in places where the model was confident about the location of a car, without increasing the number of false positives.

As a result, I implemented an algorithm that uses 64, 96, and 128 pixel size windows over different sections of the image. The 64-pixel windows are run on the image between the y-values of [400, 550], the 96-pixel windows are run over [400-650], and the 128-pixel images are run over [400, max]. Here is an image of this algorithm run over the test images. We can see that this algorithm detects cars very strongly on the test images, and gave us nice bounding boxes and strong heatmaps where the cars appear. Setting a heatmap threshold to 2 eliminated the false positives in the image.

![model_test_images][model_test_images]

### Video Implementation

Adapting the algorithm to video, I decided to add a buffer of the heatmaps to smooth the bounding boxes over frames. While the walkthrough video suggested tracking individual vehicles, I considered this to introduce additional problems, such as having to decide which bounding box corresponded to which vehicle, and having to decide when new vehicles appeared, disappeared, or were valid or invalid. Instead, I took an average of the heatmaps over the previous 10 frames. This was done after having removed bounding boxes that didn't reach a threshold set to 2. This can be seen in my `process_image` function in `main.py`.

Here's a [link to my video result](./project-mean-10frames-2threshold-3windows.mp4)

### Discussion

My project did reasonably well eliminating false-positives, although there were still some false-positives that appeared when the car passed through some areas of the video. In some areas, the white car was not detected over many frames. This indicates that the classifier could be improved, potentially through training on more data similar to the areas in the video that failed.

Processing the video also took quite a long time, given that we used sliding windows of 3 different sizes. Each frame required processing 1071 windows, which took a little over 5 seconds on my computer. I implemented the subsampling algorithm, but did not use it to generate the final video. Using the subsampling approach would improve the processing time.

Given more time, I would use a subsampling approach, and continue to tweak the algorithm to attempt to better detect vehicles in areas with which the current algorithm had trouble. A non-linear SVM could be trained that might be an improvement. In addition, I think that the bounding box calculation and tracking is an area that should be focused on. If we detect a car with fairly high confidence in a few frames, we can be fairly certain that the car will be there in subsequent frames, or at least continue upon a similar trajectory, have a similar size, etc. We can do additional calculations to improve the detection of the car and have a more accurate detection of the borders of the boxes around each car.



