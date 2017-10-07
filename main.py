from detect import *
import helpers
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.externals import joblib
import matplotlib.pyplot as plt
import glob
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
from IPython.display import HTML

TRAIN = False

# Recommended settings
settings = {
    'color_space': 'YCrCb', # RGB, HSV, LUV, HLS, YUV, YCrCb
    'orient': 9, #
    'pix_per_cell': 8,
    'cell_per_block': 2, # helps with normalization and shadows
    'hog_channel': 'ALL',
    'spatial_size': (32, 32), # spatial binning dimensions
    'hist_bins': 32,
    'spatial_feat': True,
    'hist_feat': True,
    'hog_feat': True,
    'vis': True
}

path = 'test_images/*'
example_images = glob.glob(path)

# sample image once on the entire image and then take each block separately

# Make a list of images to read in
cars = helpers.read_filenames_into_array('vehicles/', write_to_file=False, filename='cars.txt')
print('Number of vehicle images found:', len(cars))

notcars = helpers.read_filenames_into_array('non-vehicles/', write_to_file=False, filename='not_cars.txt')
print('Number of non-vehicle images found:', len(notcars))

# helpers.visualize_car_notcar(cars, notcars, settings, 'output_images/car_notcar_visualization_YCrCb_channel0.png')

# helpers.visualize_bin_spatial(cars, notcars, (32, 32), 'output_images/bin_spatial_2.png')

# helpers.visualize_color_hist(example_images[0], 'LUV', ['L', 'U', 'V'], 32, (0, 255), 'output_images/color_hist_LUV.png')

if TRAIN:
    t=time.time()
    n_samples = 1000
    random_idxs = np.random.randint(0, len(cars), n_samples)
    test_cars = cars#np.array(cars)[random_idxs]
    test_notcars = notcars#np.array(notcars)[random_idxs]

    car_features = extract_features(test_cars, color_space=settings['color_space'], spatial_size=settings['spatial_size'], hist_bins=settings['hist_bins'], orient=settings['orient'], pix_per_cell=settings['pix_per_cell'], cell_per_block=settings['cell_per_block'], hog_channel=settings['hog_channel'], spatial_feat=settings['spatial_feat'], hist_feat=settings['hist_feat'], hog_feat=settings['hog_feat'])

    notcar_features = extract_features(test_notcars, color_space=settings['color_space'], spatial_size=settings['spatial_size'], hist_bins=settings['hist_bins'], orient=settings['orient'], pix_per_cell=settings['pix_per_cell'], cell_per_block=settings['cell_per_block'], hog_channel=settings['hog_channel'], spatial_feat=settings['spatial_feat'], hist_feat=settings['hist_feat'], hog_feat=settings['hog_feat'])

    print('Seconds to compute features:\t', time.time()-t)

    X = np.vstack((car_features, notcar_features)).astype(np.float64)
    X_scaler = StandardScaler().fit(X)
    scaled_X = X_scaler.transform(X)

    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))
    rand_state = np.random.randint(0, 100)

    X_train, X_test, y_train, y_test = train_test_split(scaled_X, y, test_size=0.1, random_state=rand_state)

    print('Using:', settings['orient'], 'orientations,', settings['pix_per_cell'], 'pix per cell,', settings['cell_per_block'], 'cells per block,', settings['hog_channel'], 'hog channel,', settings['hist_bins'], 'histogram bins, and', settings['spatial_size'], 'spatial sampling.')

    print('Number of samples:\t', len(X_train))
    print('Feature vector length:\t', len(X_train[0]))

    svc = LinearSVC()
    t = time.time()
    svc.fit(X_train, y_train)

    print('Seconds to train SVC:\t', round(time.time()-t, 2))

    print('Test accurancy of SVC:\t', round(svc.score(X_test, y_test), 4))

    joblib.dump(svc, 'model-YCrCb.pkl')
    joblib.dump(X_scaler, 'scaler.pkl')
else:
    svc = joblib.load('model-YCrCb.pkl')
    X_scaler = joblib.load('scaler.pkl')

# ----------------------------------------------------------------------------

def find_cars_using_sliding_windows():
    images = []
    titles = []
    # y_start_stop = [[400, 550], [400, 550], [400, 650], [400, None]] # Min and max in y to search in slide_window()
    y_start_stop = [[400, 550], [400, 650], [400, None]]
    overlap = 0.75
    # window_size = [32, 64, 96, 128]
    window_size = [64, 96, 128]

    for img_src in example_images:
        t1 = time.time()
        img = mpimg.imread(img_src)

        heat = np.zeros_like(img[:,:,0]).astype(np.float)
        draw_image = np.copy(img)
        bbox_img = np.copy(img)
        img = img.astype(np.float32)/255

        hot_windows = []
        num_windows = 0

        for index, size in enumerate(window_size):
            windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop[index], xy_window=(size, size), xy_overlap=(overlap, overlap))

            new_hot_windows = search_windows(img, windows, svc, X_scaler, color_space=settings['color_space'], spatial_size=settings['spatial_size'], hist_bins=settings['hist_bins'], orient=settings['orient'], pix_per_cell=settings['pix_per_cell'], cell_per_block=settings['cell_per_block'], hog_channel=settings['hog_channel'], spatial_feat=settings['spatial_feat'], hist_feat=settings['hist_feat'], hog_feat=settings['hog_feat'])

            hot_windows = hot_windows + new_hot_windows
            num_windows = num_windows + len(windows)
            print('window size', size)
            print('number of hot windows', len(hot_windows))

        window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

        heat = add_heat(heat, hot_windows)
        heat = apply_threshold(heat, 2)
        heatmap = np.clip(heat, 0, 255)

        labels = label(heatmap)
        print(labels[1], 'cars found')

        bbox_img = draw_labeled_bboxes(bbox_img, labels)

        images.append(window_img)
        images.append(bbox_img)
        images.append(heatmap)
        titles.append(img_src)
        titles.append('Bounded Box')
        titles.append('Heat Map')
        print(time.time()-t1, 'seconds to process one image searching', num_windows, 'windows')

    helpers.visualize(plt.figure(figsize=(18,24)), 6, 3, images, titles, 'output_images/test_images-todelete.png')

def find_cars_using_subsampling():
    img = mpimg.imread('test_images/test1.jpg')
    ystart = 400
    ystop = 656
    scale = 1.5

    outimg = find_cars(img, ystart, ystop, scale, svc, X_scaler, settings['orient'], settings['pix_per_cell'], settings['cell_per_block'], settings['spatial_size'], settings['hist_bins'])

    plt.imshow(outimg)
    plt.savefig('subsample.jpg')

# find_cars_using_subsampling()
# find_cars_using_sliding_windows()

# -----------------------------------------


y_start_stop = [[400, 550], [400, 650], [400, None]] # Min and max in y to search in slide_window()
overlap = 0.75
window_size = [64, 96, 128]
heatmap_buffer = []

def find_cars_heatmap(img):
    heat = np.zeros_like(img[:,:,0]).astype(np.float)
    # draw_image = np.copy(img)

    img = img.astype(np.float32)/255

    hot_windows = []

    for index, size in enumerate(window_size):
        windows = slide_window(img, x_start_stop=[None, None], y_start_stop=y_start_stop[index], xy_window=(size, size), xy_overlap=(overlap, overlap))

        new_hot_windows = search_windows(img, windows, svc, X_scaler, color_space=settings['color_space'], spatial_size=settings['spatial_size'], hist_bins=settings['hist_bins'], orient=settings['orient'], pix_per_cell=settings['pix_per_cell'], cell_per_block=settings['cell_per_block'], hog_channel=settings['hog_channel'], spatial_feat=settings['spatial_feat'], hist_feat=settings['hist_feat'], hog_feat=settings['hog_feat'])

        hot_windows = hot_windows + new_hot_windows

    # window_img = draw_boxes(draw_image, hot_windows, color=(0, 0, 255), thick=6)

    heat = add_heat(heat, hot_windows)
    heat = apply_threshold(heat, 2)
    heatmap = np.clip(heat, 0, 255)

    return heatmap

def process_image(img):
    draw_image = np.copy(img)

    new_heatmap = find_cars_heatmap(img)

    heatmap_buffer.append(new_heatmap)
    if (len(heatmap_buffer) > 10):
        heatmap_buffer.pop(0)

    heatmap = np.mean(heatmap_buffer, axis=0)

    labels = label(heatmap)
    draw_image = draw_labeled_bboxes(draw_image, labels)
    return draw_image

# test_output = 'test-mean-sdafdaf.mp4'
# clip = VideoFileClip('test_video.mp4')
# # clip = VideoFileClip('project_video.mp4')
# test_clip = clip.fl_image(process_image)
# test_clip.write_videofile(test_output, audio=False)

# class Vehicle():
#     def __init__(self):
#         self.detected = False # was the vehicle detected in the last iteration

#         self.n_detections = 0 # number of times this vehicle has been detected

#         self.n_nondetections = 0 # number of times this vehicle has not been detected

#         self.xpixels = None # Pixel x values of last detection
#         self.ypixels = None # Pixel y values of last detection

#         self.recent_xfitted = [] # x position of he last n fits of the bounding box

#         self.bestx = None # average x position of the last n fits

#         self.recent_yfitted = [] # y position of the last n fits of the bounding box

#         self.besty = None # average y position of the last n fits

#         self.recent_wfitted = [] # width of the last n fits of the bounding box

#         self.bestw = None # average width of the last n fits

#         self.recent_hfitted = [] # height of the last n fits of the bounding box

#         self.besth = None # average height of the last n fits


# a = np.array([[1, 2, 3, 4], [5, 6, 7, 8]])
# print(a)
# print(a.shape)
# b = np.array([[10, 20, 30, 40], [50, 60, 70, 80]])
# print(b)
# print(b.shape)

# c = np.sum([a, b], axis=0)
# print(c)