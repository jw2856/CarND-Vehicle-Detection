import os
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from detect import *

def read_filenames_into_array(base_dir, write_to_file=False, filename=None):
    subdirectories = os.listdir(base_dir)

    output = []
    for s in subdirectories:
        output.extend(glob.glob(base_dir+s+'/*'))

    if write_to_file:
        if filename == None:
            raise Exception('You must provide a filename to write to file')

        with open(filename, 'w') as f:
            for fn in output:
                f.write(fn+'\n')

    return output

def visualize(fig, rows, cols, imgs, titles, filename):
    for i, img in enumerate(imgs):
        plt.subplot(rows, cols, i+1)
        plt.title(titles[i])

        img_dims = len(img.shape)
        if img_dims < 3:
            plt.imshow(img, cmap='hot')
        else:
            plt.imshow(img)
    
    plt.savefig(filename)

def visualize_bin_spatial(cars, not_cars, size, filename):
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(not_cars))

    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(not_cars[notcar_ind])

    car_features = cv2.resize(car_image, size)
    notcar_features = cv2. resize(notcar_image, size)

    images = [car_image, car_features, notcar_image, notcar_features]
    titles = ['Car', 'Spatially Binned', 'Not Car', 'Spatially Binned']
    visualize(plt.figure(figsize=(12,3)), 1, 4, images, titles, filename)

def visualize_color_hist(img, color_space, channel_names, bins, range, filename):
    car_image = mpimg.imread(img)

    if color_space != 'RGB':
        if color_space == 'HSV':
            feat_car_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            feat_car_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            feat_car_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            feat_car_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            feat_car_image = cv2.cvtColor(car_image, cv2.COLOR_RGB2YCrCb)
    else: feat_car_image = np.copy(car_image)

    # Compute the histogram of the RGB channels separately
    car_rhist = np.histogram(feat_car_image[:,:,0], bins=bins, range=range)
    car_ghist = np.histogram(feat_car_image[:,:,1], bins=bins, range=range)
    car_bhist = np.histogram(feat_car_image[:,:,2], bins=bins, range=range)
    # Generating bin centers
    car_bin_edges = car_rhist[1]
    car_bin_centers = (car_bin_edges[1:]  + car_bin_edges[0:len(car_bin_edges)-1])/2

    fig = plt.figure(figsize=(12,3))

    plt.subplot(1, 4, 1)
    plt.imshow(feat_car_image)

    plt.subplot(1, 4, 2)
    plt.bar(car_bin_centers, car_rhist[0])
    plt.xlim(0, 255)
    plt.title(channel_names[0])
    plt.subplot(1, 4, 3)
    plt.bar(car_bin_centers, car_ghist[0])
    plt.xlim(0, 255)
    plt.title(channel_names[1])
    plt.subplot(1, 4, 4)
    plt.bar(car_bin_centers, car_bhist[0])
    plt.xlim(0, 255)
    plt.title(channel_names[2])
    fig.tight_layout()
    fig.savefig(filename)


def visualize_car_notcar(cars, not_cars, settings, filename):
    car_ind = np.random.randint(0, len(cars))
    notcar_ind = np.random.randint(0, len(not_cars))

    car_image = mpimg.imread(cars[car_ind])
    notcar_image = mpimg.imread(not_cars[notcar_ind])

    car_features, car_hog_image = single_img_features(car_image, color_space=settings['color_space'], spatial_size=settings['spatial_size'], hist_bins=settings['hist_bins'], orient=settings['orient'], pix_per_cell=settings['pix_per_cell'], cell_per_block=settings['cell_per_block'], hog_channel=settings['hog_channel'], spatial_feat=settings['spatial_feat'], hist_feat=settings['hist_feat'], hog_feat=settings['hog_feat'], vis=settings['vis'])

    notcar_features, notcar_hog_image = single_img_features(notcar_image, color_space=settings['color_space'], spatial_size=settings['spatial_size'], hist_bins=settings['hist_bins'], orient=settings['orient'], pix_per_cell=settings['pix_per_cell'], cell_per_block=settings['cell_per_block'], hog_channel=settings['hog_channel'], spatial_feat=settings['spatial_feat'], hist_feat=settings['hist_feat'], hog_feat=settings['hog_feat'], vis=settings['vis'])

    images = [car_image, car_hog_image, notcar_image, notcar_hog_image]
    titles = ['Car', 'Car HOG', 'Not Car', 'Not Car HOG']
    visualize(plt.figure(figsize=(12,3)), 1, 4, images, titles, filename)