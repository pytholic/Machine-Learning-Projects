import cv2
import numpy as np
import os

#create lists to save the labels (the name of the shape)
train_labels, train_images = [],[]
train_dir = './abcde'
shape_list = ['a', 'b', 'c', 'd', 'e']

#function to preprocess data
def preprocess(images, labels):
    """You can make your preprocessing code in this function.
    Here, we just flatten the images, for example.
    In addition, you can split this data into the training set and validation set for robustness to the test(unseen) data.

    :params list images: (Number of images x row x column)
    :params list labels: (Number of images, 1)
    :rtype: array
    :return: preprocessed images and labels
    """
    dataDim = np.prod(images[0].shape)
    images = np.array(images)
    images = images.reshape(len(images), dataDim)
    images = images.astype('float32')
    images /=255
    labels = np.array(labels)
    return images, labels


# function to make classifier
def classify(images, labels):
    """You can make your classifier code in this function.
    Here, we use KNN classifier, for example.

    :params array images: (Number of images x row x column)
    :params array labels: (Number of images)
    :return: classifier model
    """
    from sklearn.neighbors import KNeighborsClassifier
    neigh = KNeighborsClassifier(n_neighbors=2)
    neigh.fit(images, labels)
    return neigh


if __name__ == '__main__':
    #iterate through each shape
    for shape in shape_list:
        print('Getting data for: ', shape)
        for file_name in os.listdir(os.path.join(train_dir,shape)):
            train_images.append(cv2.imread(os.path.join(train_dir,shape,file_name), 0))
            #add an integer to the labels list
            train_labels.append(shape_list.index(shape))

    print('Number of training images: ', len(train_images))

    # Preprocess (your own function)
    train_images, train_labels = preprocess(train_images, train_labels)

    # Make a classifier (your own function)
    model = classify(train_images, train_labels)

    # Predict the labels from the model (your own code depending the output of the train function)
    pred_labels = model.predict(train_images)

    # Calculate accuracy (Do not erase or modify here)
    pred_acc = np.sum(pred_labels==train_labels)/len(train_labels)*100
    print("Accuracy = {}".format(pred_acc))


    """forTA (You can modify the below code depending on your above code. If you do, the TAs will be much more convenient to grade.)

    test_dir = '../ForTA/abcde'
    test_labels, test_images = [], []
    for shape in shape_list:
        print('Getting data for: ', shape)
        for file_name in os.listdir(os.path.join(test_dir,shape)):
            test_images.append(cv2.imread(os.path.join(test_dir,shape,file_name), 0))
            #add an integer to the labels list
            test_labels.append(shape_list.index(shape))

    print('Number of test images: ', len(test_images))

    test_images, test_labels = preprocess(test_images, test_labels)
    pred_labels = model.predict(test_images)
    pred_acc = np.sum(pred_labels==test_labels)/len(test_labels)*100
    print("Test Accuracy = {}".format(pred_acc))
    """