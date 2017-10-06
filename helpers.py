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