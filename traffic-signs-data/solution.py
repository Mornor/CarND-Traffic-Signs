# Load pickled data
import pickle
import numpy as np

# Fill this in based on where you saved the training and testing data

training_file = 'traffic-signs-data/train.p'
testing_file = 'traffic-signs-data/test.p'

with open(training_file, mode='rb') as f:
    train = pickle.load(f)
with open(testing_file, mode='rb') as f:
    test = pickle.load(f)

X_train, y_train = train['features'], train['labels']
X_test, y_test = test['features'], test['labels']

# Number of training examples
n_train = len(X_train)

# Number of testing examples.
n_test = len(X_test)

# What's the shape of an traffic sign image?
image_shape = 1024  # 32*32 pixels

# How many unique classes/labels there are in the dataset.
# See numpy.unique : https://docs.scipy.org/doc/numpy/reference/generated/numpy.unique.html
unique_items, counts = np.unique(y_train, return_counts=True)
n_classes = len(counts)

print("Number of training examples =", n_train)
print("Number of testing examples =", n_test)
print("Image data shape =", image_shape)
print("Number of classes =", n_classes)

import matplotlib.pyplot as plt
import csv


# Visualizations will be shown in the notebook.

# Create figure with 3x3 sub-plots.
def plot_images(images, classes, labels):
    fig, axes = plt.subplots(3, 3)
    fig.subplots_adjust(hspace=0.3, wspace=0.3)

    for i, ax in enumerate(axes.flat):
        # Plot image.
        ax.imshow(images[i], cmap='gray')
        xlabel = labels[classes[i]][1]
        ax.set_xlabel(xlabel)

        # Remove pixels scale legend x and legend y.
        ax.set_xticks([])
        ax.set_yticks([])


def read_csv(path):
    sign_names = []
    with open(path, 'r') as f:
        reader = csv.reader(f)
        f.readline()  # Skip first line
        for row in reader:
            sign_names.append(row)
    return sign_names


labels = read_csv('signnames.csv')
plot_images(X_train, y_train, labels)
plt.show()
