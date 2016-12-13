# Self-Driving Car Engineer Nanodegree
# Deep Learning
## Project: Build a Traffic Sign Recognition Program

### Overview

This is the 2nd project of the <a href="https://www.udacity.com/course/self-driving-car-engineer-nanodegree--nd013">Self Driving Car Engineer Nanodegree</a> I am taking part. <br>
In this project, I use a deep neural networks (convolutional neural networks) to classify traffic signs. The trained model can decode traffic signs from images of the [German Traffic Sign Dataset](http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset). <br>
The model is then test on new images of traffic signs. 

### Model
I decided to take as a starter the article from Pr. Yann LeCunn, which you can found [here](http://yann.lecun.com/exdb/publis/pdf/sermanet-ijcnn-11.pdf). 
My model observe the following structure: 
- <b>First Layer</b>: Convolutional layer, output of size 32x32x6
- <b>Max pooling layer</b> Output of size 14x14x6
- <b>Second layer</b> Convolutional layer, output of size 10x10x16
- <b>Max pooling layer</b> Output of size 5x5x6
- <b>Flatten Layer</b> Flatten shape to 1D
- <b>Fully connected layer 1</b> Output of size 120
- <b>Fully connected layer 2</b> Output of size n_classes (43 in that case)

### Dependencies

This project requires **Python 3.5** and the following Python libraries installed:

- [Jupyter](http://jupyter.org/)
- [NumPy](http://www.numpy.org/)
- [SciPy](https://www.scipy.org/)
- [scikit-learn](http://scikit-learn.org/)
- [TensorFlow](http://tensorflow.org)
- [Matplotlib](http://matplotlib.org/)
- [OpenCV](http://opencv.org/)

### Dataset

1. [Download the dataset](https://d17h27t6h515a5.cloudfront.net/topher/2016/November/581faac4_traffic-signs-data/traffic-signs-data.zip). This is a pickled dataset in which we've already resized the images to 32x32.

2. Start the notebook.
```
jupyter notebook Traffic_Signs_Recognition.ipynb
```

#### Notes

Choose a relative small batch size as well as few epochs if you want to run it locally, since the model demands heavy computations.  



