# This file includes various feature functions like spatial binning, color histogram and hog as shown in lectures

import cv2
import numpy as np
from skimage.feature import hog

class featureFunctions(object):

    def __init__(self):

        self.spatial_features = None
        self.hist_features = None
        self.hog_features =  None

    # A function to compute binned color features
    def spatial_binned(self,image, size=(32, 32)):
        # Use cv2.resize().ravel() to create the feature vector

        color1 = cv2.resize(image[:, :, 0], size).ravel()
        color2 = cv2.resize(image[:, :, 1], size).ravel()
        color3 = cv2.resize(image[:, :, 2], size).ravel()
        self.spatial_features = np.hstack((color1, color2, color3))
        return self.spatial_features

    # A function to compute color histogram features
    def color_hist(self,image, nbins=32, bin_range=(0, 256)):
        # np.histogram() returns a tuple of two arrays. item one contains the counts in each of the bins
        # and other contains the bin edges( it will be one element longer than the item one)
        channel1_hist = np.histogram(image[:, :, 0], bins=nbins, range=bin_range)
        channel2_hist = np.histogram(image[:, :, 1], bins=nbins, range=bin_range)
        channel3_hist = np.histogram(image[:, :, 2], bins=nbins, range=bin_range)
        self.hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
        return self.hist_features

    def get_hog_features(self, image,
                         orientation = 9,
                         pixels_per_cell = 8,
                         cell_per_block = 2,
                         visualize = True,
                         feature_vec = True):

        self.hog_features, self.hog_image = hog(image, orientations=orientation, pixels_per_cell=(pixels_per_cell, pixels_per_cell),
                                                cells_per_block=(cell_per_block, cell_per_block), transform_sqrt=True,
                                                visualise = True,
                                                feature_vector= feature_vec)


        if visualize:
            return self.hog_features, self.hog_image
        else:
            return self.hog_features

