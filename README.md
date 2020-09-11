# ud_image_classifier
### Deep Learning

Going forward, AI algorithms will be incorporated into more and more everyday applications. For example, we might want to include an image classifier in a smart phone app. To do this, we would use a deep learning model trained on hundreds of thousands of images as part of the overall application architecture. A large part of software development in the future will be using these types of models as common parts of applications. 

In this project, we will train an image classifier to recognize different species of flowers. Imagine using something like this in a phone app that tells the name of the flower the camera is looking at. In practice we will train this classifier, then export it for use in an application. We'll be using [this dataset](http://www.robots.ox.ac.uk/~vgg/data/flowers/102/index.html) of 102 flower categories.

The project is broken down into multiple steps:

* Load and preprocess the image dataset
* Train the image classifier on the dataset
* Use the trained classifier to predict image content


In the end we will have an application that can be trained on any set of labeled images. Here the network will be learning about flowers and end up as a command line application.
