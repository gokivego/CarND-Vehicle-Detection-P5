import os
import glob
# %matplotlib inline
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2
import time
import pickle
from skimage.feature import hog
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from scipy.ndimage.measurements import label
# from moviepy.editor import VideoFileClip
# from IPython.display import HTML
from loadImages import *
from featureFunctions import featureFunctions
import extractor
from drawWindows import *
import classifier
from heatMap import *
from classifier import classifier


# Load car and not car images

cars = loadImages("./vehicles/", "cars")
notcars = loadImages("./non-vehicles/", "notcars")

# We can save the trained classifier in a .pkl file and then retrieve it to train the classifier

if os.path.isfile("models/trained_model.p"):
    print("Model already present, retrieving to use it")
    train_model = False
else:
    print("Training a classifier using new images")
    train_model = True

# Train a classifier

c = classifier(cars,notcars,train_model)
c.classify()