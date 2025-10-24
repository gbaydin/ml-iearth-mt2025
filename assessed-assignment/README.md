## Assessed assignment

Assessment of this module will be based on an assignment where each student implements a neural network (NN) from scratch, without using the pre-implemented NN modules in popular machine learning frameworks such as PyTorch and TensorFlow, and trains it to perform regression or classification with a small dataset of their choice.

The components needed are:

- The NN architecture "from scratch". This means to implement, as a minimum, a fully-connected NN layer where the layer's output is the matrix multiplication of the input with the layer's weight matrix, plus a bias term. The network should ideally be composed of at least two such fully-connected layers (i.e., a multi-layer Perceptron), and some nonlinear activation function (e.g., ReLU, tanh). **Using pre-implemented NN modules such as torch.nn.Linear in PyTorch or tensorflow.layers.dense in TensorFlow is not allowed.** You are free to implement this using either pure Python lists of scalars (you would need to implement some basic linear algebra), Numpy arrays, PyTorch or TensorFlow tensor types.

- A choice of NN weight initialization, e.g., [Xavier & Bengio (2010)](http://proceedings.mlr.press/v9/glorot10a/glorot10a.pdf), [He et al. (2015)](https://arxiv.org/abs/1502.01852). Hint: you can see formulae for some options [here](https://pytorch.org/docs/stable/nn.init.html).

- A small dataset such as [MNIST](http://yann.lecun.com/exdb/mnist/), [Boston housing dataset](https://www.kaggle.com/schirmerchad/bostonhoustingmlnd), [Iris flower dataset](https://www.kaggle.com/arshid/iris-flower-dataset), etc.

- A choice of loss function suitable for the regression or classification task you would define between some chosen features in your dataset. E.g., [mean squared error](https://en.wikipedia.org/wiki/Mean_squared_error) for regression or [cross-entropy](https://en.wikipedia.org/wiki/Cross_entropy) for classification.

- The gradient-based optimization loop, where you feed a data minibatch to the NN, compute the output, compute the loss, take the derivative of the loss with respect to NN weights and update the weights via gradient-descent update rule (e.g., SGD, SGD with momentum, Adam) that you manually implement. **Using pre-implemented optimizers such as torch.optim.Adam is not allowed.** In order to compute the gradients, you are free to use the automatic differentiation capabilities of the tensor type you use (e.g., use autograd functionality in PyTorch, or use [JAX](https://jax.readthedocs.io/en/latest/notebooks/quickstart.html) or [autograd](https://github.com/HIPS/autograd) if you're using Numpy arrays.)

If you are interested in a more challenging case, and in gaining a deeper understanding of "under-the-hood" working of ML frameworks, you may also deliver the following:

- Manually-implemented derivative code in terms of a minimal autodiff approach such as https://github.com/karpathy/micrograd or https://github.com/mattjj/autodidact and then using the differentiable type (scalar for micrograd and differentiable Numpy array for autodidact) as a basis for your NN code and gradients.
- A more complicated architecture such as a recurrent neural network (RNN) (e.g., LSTM, GRU), or a convolutional neural network (CNN) (e.g., 2d convolutions, pooling). For example an LSTM can be implemented as a collection of fully-connected layers and nonlinearities or convolutional NN layers can be implemented in terms of matrix multiplication operations.

### Choice of programming language

Python is the preferred programming language for this assignment, but you can use other languages as well.
