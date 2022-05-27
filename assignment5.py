from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import mahotas
import cv2
import os
import h5py
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.model_selection import KFold
import time
import matplotlib.pyplot as plt

image_size = tuple((500,500))
h5_data = "a5_output/data.h5"
h5_labels = "a5_output/labels.h5"

train_labels = ['hot_dog', 'not_hot_dog']
global_features = []
labels = []

def get_color_histogram(image):
    #Converts the image to HSV
    image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    #Get the histogram and normalize it
    hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist, hist)
    hist = hist.flatten()
    return hist

def get_hu_moments(image):
    #Converts the image to greyscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #Gets the Hu Moments as a flattened list
    feature = cv2.HuMoments(cv2.moments(image)).flatten()
    return feature

def get_haralick_textures(image):
    #Converts the image to greyscale
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    haralick = mahotas.features.haralick(image).mean(axis=0)
    return haralick

def process_images():
    for tl in train_labels:
        dir = 'hotdog/train/' + tl + '/'
        image_list = os.listdir(dir)
        for img in image_list:
            image = cv2.imread(dir + img)
            img_size = tuple((500, 500))
            image = cv2.resize(image, img_size)

            histogram = get_color_histogram(image)
            shape = get_hu_moments(image)
            textures = get_haralick_textures(image)

            gf = np.hstack([histogram, shape, textures])
            labels.append(tl)
            global_features.append(gf)
        print(tl  + " processed")

def encode_and_save():
    le = LabelEncoder()
    target = le.fit_transform(labels)
    scaler = MinMaxScaler(feature_range=(0, 1))
    rescaled_features = scaler.fit_transform(global_features)
    h5f_data = h5py.File(h5_data, 'w')
    h5f_data.create_dataset('dataset_1', data=np.array(rescaled_features))

    h5f_label = h5py.File(h5_labels, 'w')
    h5f_label.create_dataset('dataset_1', data=np.array(target))

    h5f_data.close()
    h5f_label.close()

def train_model():
    results = []
    names = []
    h5f_data  = h5py.File(h5_data, 'r')
    h5f_label = h5py.File(h5_labels, 'r')
    global_features_string = h5f_data['dataset_1']
    global_labels_string = h5f_label['dataset_1']
    global_features = np.array(global_features_string)
    global_labels   = np.array(global_labels_string)
    h5f_data.close()
    h5f_label.close()
    (trainData, testData, trainLabels, testLabels) = train_test_split(np.array(global_features),
                                                                                          np.array(global_labels),
                                                                                          test_size=0.1,
                                                                                          random_state=9)
    kfold = KFold(n_splits=10)
    cv_results = cross_val_score(KNeighborsClassifier(), trainData, trainLabels, cv=kfold, scoring='accuracy')
    msg = "%s: %f (%f)" % ('KNN: ', cv_results.mean(), cv_results.std())
    print(msg)

    clf = KNeighborsClassifier()
    clf.fit(trainData, trainLabels)

    for tl in train_labels:
        dir = 'hotdog/train/' + tl + '/'
        image_list = os.listdir(dir)
        for img in image_list:
            image = cv2.imread(dir + img)
            img_size = tuple((500, 500))
            image = cv2.resize(image, img_size)

            histogram = np.array(get_color_histogram(image))
            shape = np.array(get_hu_moments(image))
            textures = np.array(get_haralick_textures(image))

            histogram = get_color_histogram(image)
            shape = get_hu_moments(image)
            textures = get_haralick_textures(image)

            gf = np.hstack([histogram, shape, textures])

            # scale features in the range (0-1)
            scaler = MinMaxScaler(feature_range=(0, 1))
            rescaled_feature = scaler.fit_transform([gf])

            prediction = clf.predict(rescaled_feature)[0]
            cv2.putText(image, prediction[0], (20,30), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,255,255), 3)

            # display the output image
            plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            plt.show()
            #time.sleep(5)

process_images()
encode_and_save()
train_model()



#print(os.listdir('hotdog/test/hot_dog'))

#print(get_hu_moments(cv2.imread(os.getcwd() + '/hotdog/test/hot_dog/133012.jpg')))
#print(get_color_histogram(cv2.imread(os.getcwd() + '/hotdog/test/hot_dog/133012.jpg')))
#print(get_haralick_textures(cv2.imread(os.getcwd() + '/hotdog/test/hot_dog/133012.jpg')))
