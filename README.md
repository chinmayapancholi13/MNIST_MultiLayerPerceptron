# MNIST_MultiLayerPerceptron
Code to predict MNIST digits by implementing a Multi Layer Perceptron with a single hidden layer, without using any libraries like Tensorflow, Theano etc. Only linear algebra libraries like numpy have been used for the implementation.

The dataset has been downloaded from http://yann.lecun.com/exdb/mnist/ (four files). The 4 files should be extracted into a folder named `data` just outside the folder containing the `main.py` file i.e. the code in the file `main.py` reads the input data files from the folder '../data'. The `train` function trains the neural network given the training examples and saves the weights in a folder named `weights` in the same folder as `main.py`. The `test` function reads the saved weights and given the test examples, it returns the predicted labels.

# Resources
1. https://ift6266h16.wordpress.com/2016/01/11/first-assignment-mlp-on-mnist/
2. https://github.com/tfjgeorge/ift6266/blob/master/notebooks/MLP.ipynb
