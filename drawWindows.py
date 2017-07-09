import numpy as np
import cv2
from featureFunctions import featureFunctions
from extractor import featureVector, featureExtractor

# A function to draw bounding boxes
def draw_bboxes(image, bboxes, color = (255,0,0), thick = 3):

    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

# Now we define a function that takes the start and stop position in an image

def slide_window(img, x_start_stop=[None, None], y_start_stop=[None, None],
                 window_size = (64, 64), window_overlap=(0.5, 0.5)):

    # We set the start and stop to image size if the start and stop positions are
    # not defined
    if x_start_stop[0] == None:
        x_start_stop[0] = 0
    if x_start_stop[1] == None:
        x_start_stop[1] = img.shape[1]
    if y_start_stop[0] == None:
        y_start_stop[0] = 0
    if y_start_stop[1] == None:
        y_start_stop[1] = img.shape[0]

    # We need to break up the image into blocks and cells.
    # For this we need the dimension of the search region
    xsearch = x_start_stop[1] - x_start_stop[0]
    ysearch = y_start_stop[1] - y_start_stop[0]
    # Compute the number of pixels per each step we take
    pix_per_step_x = np.int(window_size[0] * (1 - window_overlap[0]))
    pix_per_step_y = np.int(window_size[1] * (1 - window_overlap[1]))
    # Compute the number of windows in x/y
    buffer_x = np.int(window_size[0] * (window_overlap[0]))
    buffer_y = np.int(window_size[1] * (window_overlap[1]))
    num_windows_x = np.int((xsearch - buffer_x) / pix_per_step_x)
    num_windows_y = np.int((ysearch - buffer_y) / pix_per_step_y)
    # Initialize a list to append window positions to
    window_list = []
    #We Loop through finding x and y window positions
    for ys in range(num_windows_y):
        for xs in range(num_windows_x):
            # Calculate window position
            startx = x_start_stop[0] + xs * pix_per_step_x
            endx = startx + window_size[0]
            starty = y_start_stop[0] + ys * pix_per_step_y
            endy = starty + window_size[1]

            # Append window position to list
            window_list.append(((startx, starty), (endx, endy)))
    # Return the list of windows
    return window_list

def search_windows(image, windows, clf, scaler, size=(32, 32), nbins = 32, orientation = 9, pixels_per_cell=8, cell_per_block=2):

    # 1) Create an empty list to receive positive detection windows
    on_windows = []
    # 2) Iterate over all windows in the list
    for window in windows:
        # 3) Extract the test window from original image
        test_img = cv2.resize(image[window[0][1]:window[1][1], window[0][0]:window[1][0]], (64, 64))
        # 4) Extract features for that window using single_img_features()
        features = featureVector()(test_img, size, nbins, orientation, pixels_per_cell, cell_per_block)
        # 5) Scale extracted features to be fed to classifier
        test_features = scaler.transform(np.array(features).reshape(1, -1))
        # 6) Predict using your classifier
        prediction = clf.predict(test_features)
        # 7) If positive (prediction == 1) then save the window
        if prediction == 1:
            on_windows.append(window)
    # 8) Return windows for positive detections
    return on_windows