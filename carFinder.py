import matplotlib.pyplot as plt
import numpy as np
import cv2
from featureFunctions import featureFunctions

class Car():
	def __init__(self):
		self.heatmap = np.array([None]*10)
		self.first_frame = True
		self.

def carFinder(image,scale, svc,
			  X_scalar, y_range=(None, None),
			  pixels_per_cell=8,
			  orientation=9,
			  cell_per_block=2,
			  size= (32,32),
			  nbins = 32):

	scale_factors = [1.3,1.5,1.8]
	draw_img = np.copy(img)
	window_count = 0
	boxes = []

	heatmap = np.zeros_like(img[:, :, 0])
	image = image.astype(np.float32) / 255

	for scale in scale_factors:

		image_to_search = image[y_range[0]:y_range[1], :, :]
		ctrans_tosearch = cv2.cvtColor(image_to_search, cv2.COLOR_RGB2YCrCb)

		if scale != 1:
			imshape = ctrans_tosearch.shape
			ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1] / scale), np.int(imshape[0] / scale)))
		ch1 = ctrans_tosearch[:, :, 0]
		ch2 = ctrans_tosearch[:, :, 1]
		ch3 = ctrans_tosearch[:, :, 2]

		# number of blocks
		nxblocks = (ch1.shape[1] // pixels_per_cell) - 1
		nyblocks = (ch2.shape[0] // pixels_per_cell) - 1

		nfeat_per_block = orientation * cell_per_block ** 2
		windows = 64

		nblocks_per_window = (windows // pixels_per_cell) - 1
		cell_per_step = 2

		nxsteps = (nxblocks - nblocks_per_window) // cell_per_step
		nysteps = (nyblocks - nblocks_per_window) // cell_per_step

		# Computing individual channel HOG features for the entire image
		hog1 = featureFunctions.get_hog_features(ch1, orientation, pixels_per_cell, cell_per_block, visualize = False, feature_vec=False)
		hog2 = featureFunctions.get_hog_features(ch2, orientation, pixels_per_cell, cell_per_block, visualize= False, feature_vec=False)
		hog3, hog3_img = featureFunctions.get_hog_features(ch3, orientation, pixels_per_cell, cell_per_block, visualize= True, feature_vec=False)

		for xb in range(nxsteps):
			for yb in range(nysteps):

				ypos = yb * cell_per_step
				xpos = xb * cell_per_step

				hog_feat1 = hog1[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
				hog_feat2 = hog2[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()
				hog_feat3 = hog3[ypos:ypos + nblocks_per_window, xpos:xpos + nblocks_per_window].ravel()

				hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
				xleft = xpos * pixels_per_cell
				ytop = ypos * pixels_per_cell

				subimg = cv2.resize(ctrans_tosearch[ytop: ytop + windows, xleft:xleft + windows], (64, 64))

				# color and spatial binning
				spatial_feature = featureFunctions.spatial_binned(subimg, size)

				hist_features = featureFunctions.color_hist(subimg, nbins)
				features = np.hstack((spatial_feature, hist_features, hog_features)).reshape(1, -1)
				test_features = X_scalar.transform(features)

				test_prediction = svc.predict(test_features)
				if test_prediction == 1:
					xbox_left = np.int(xleft * scale)
					ytop_draw = np.int(ytop * scale)
					win_draw = np.int(windows * scale)
					print(ytop_draw, y_range[0])
					cv2.rectangle(draw_img, (xbox_left, ytop_draw + y_range[0]),
								  (xbox_left + win_draw, ytop_draw + win_draw + y_range[0]), (0, 0, 255), 6)
					boxes.append(((xbox_left, ytop_draw + y_range[0]),
									  (xbox_left + win_draw, ytop_draw + win_draw + y_range[0])))
					heatmap[ytop_draw + y_range[0]:ytop_draw + win_draw + y_range[0], xbox_left: xbox_left + win_draw] += 1

				window_count += 1

		# if vehicle_detected.first_frame ==

	return draw_img, heatmap