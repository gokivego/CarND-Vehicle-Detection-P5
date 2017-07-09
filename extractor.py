# Extractor file includes classes to extract features from images

import cv2
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from featureFunctions import featureFunctions

# This class is used to derive feature vector from a single image
class featureVector(object):

    def __init__(self):
        pass

    def __call__(self, image, size = (32,32), nbins = 32, orientation = 9, pixels_per_cell = 8,
                 cell_per_block = 2, visualize = False):

        # We convert the image to YCrCb space first

        image = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
        image_features = []

        image_features.append(featureFunctions.spatial_binned(image))
        image_features.append(featureFunctions.color_hist(image))

        hog_features  = []
        for channel in range(image.shape[2]):
            hog_features.extend(featureFunctions.get_hog_features(image[:,:,channel], visualize= False))

        hog_features = np.ravel(hog_features)
        image_features.append(hog_features)
        return np.concatenate(image_features)

# To extract features from a set of image files

class featureExtractor(object):

    def __init__(self):

        self.features = []

    def get_features_from_images(self, images, size = (32,32), nbins = 32, orientation = 9,
                                 pixels_per_cell = 8 , cell_per_block = 2):

        for file in images:
            image_features = []
            image = mpimg.imread(file)
            image_feat = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)

            fF = featureFunctions()
            spatial_features = fF.spatial_binned(image_feat, size)
            image_features.append(spatial_features)

            hist_features = fF.color_hist(image_feat, nbins)
            image_features.append(hist_features)

            hog_features = []
            for channel in range(image_feat.shape[2]):
                hog_features.extend(fF.get_hog_features(image_feat[:,:,channel], orientation, pixels_per_cell, cell_per_block,
                                                             visualize= False, feature_vec= True))
            hog_features = np.ravel(hog_features)

            image_features.append(hog_features)

            self.features.append(np.concatenate(image_features))

        return self.features





