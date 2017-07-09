import numpy as np
from extractor import featureExtractor, featureVector
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
import pickle


class classifier(object):

    def __init__(self, cars, notcars, train_model = True, size=(32,32), nbins = 32, orientation = 9, pixels_per_cell = 8, cell_per_block = 2):

        self.cars = cars
        self.notcars = notcars
        self.train_model = train_model
        self.size = size
        self.nbins = nbins
        self.orientation = orientation
        self.pixels_per_cell = pixels_per_cell
        self.cell_per_block = cell_per_block

    def extract_features_from_images(self):
        if self.train_model:
            num_samples = 10000
        else:
            num_samples = 10

        # Take random samples from the images to train the model
        random_idxs = np.random.randint(0, len(self.cars), num_samples)

        test_cars = np.array(self.cars)[random_idxs]
        test_notcars = np.array(self.notcars)[random_idxs]

        print ("Extracting Car Features")
        fE_cars = featureExtractor()
        car_features = fE_cars.get_features_from_images(test_cars, self.size, self.nbins,
                                                   self.orientation,self.pixels_per_cell,
                                                   self.cell_per_block)

        print("Extracting Non Car Features")
        fE_notcars = featureExtractor()
        not_car_features = fE_notcars.get_features_from_images(test_notcars, self.size, self.nbins,
                                                       self.orientation,self.pixels_per_cell,
                                                       self.cell_per_block)

        return car_features, not_car_features

    def get_train_test_data(self):

        car_feat, not_car_feat = self.extract_features_from_images()

        pickle.dump(car_feat, open("./models/car_feat.p", "wb"))
        pickle.dump(not_car_feat, open("./models/not_car_feat.p", "wb"))

        X = np.vstack((car_feat, not_car_feat)).astype(np.float64)
        X_scalar = StandardScaler().fit(X)
        scaledX = X_scalar.transform(X)

        pickle.dump(X_scalar,open('./models/X_scalar.p',"wb"))


        y = np.hstack((np.ones(len(car_feat)), np.zeros(len(not_car_feat))))
        rand_state = np.random.randint(0, 100)

        if self.train_model:
            split_size = 0.1
        else:
            split_size = 0.9

        X_train, X_test, y_train, y_test = train_test_split(scaledX, y, test_size=split_size, random_state=rand_state)

        return X_train, X_test, y_train, y_test

    def classify(self):

        self.X_train, self.X_test, self.y_train, self.y_test = self.get_train_test_data()

        if self.train_model:
            print ("Training Model")
            svc = LinearSVC()
            svc.fit(self.X_train, self.y_train)
            print ("Finished Training Model")
            acc = round(svc.score(self.X_test, self.y_test), 4)
            print("Test accuracy is", acc)

            # save the model here
            train_model_file = "./models/trained_model.p"
            output = open(train_model_file, 'wb')
            pickle.dump(svc, output)
            output.close()

        else:
            trained_model_file = "./models/trained_model.p"
            with open(trained_model_file, mode='rb') as f:
                svc_X_scalar = pickle.load(f)
                svc = svc_X_scalar["svc"]

            acc = round(svc.score(self.X_test, self.y_test), 4)
            print("Test accuracy is", acc)
