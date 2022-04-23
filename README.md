# Deep-Learning-with-Pytorch
Trying To implement some of examples and exercises in Pytorch to review my knowledge
Tensors are the PyTorch equivalent to Numpy arrays. A vector is a 1-D tensor, and a matrix a 2-D tensor. When working with neural networks, we will use tensors of various shapes and number of dimensions.
torch.zeros: Creates a tensor filled with zeros
torch.ones: Creates a tensor filled with ones
torch.rand: Creates a tensor with random values uniformly sampled between 0 and 1
torch.randn: Creates a tensor with random values sampled from a normal distribution with mean 0 and variance 1
torch.arange: Creates a tensor containing the values N,N+1,N+2,...,M
torch.Tensor (input list): Creates a tensor from the list elements you provide
obtain the shape of a tensor in the same way as in numpy (x.shape), or using the .size method.
Converting Tensor to Numpy, and Numpy to Tensor
To transform a numpy array into a tensor, we can use the function torch.from_numpy.
To transform a PyTorch tensor back to a numpy array, we can use the function .numpy() on tensors.
Operations by Pytorch
Indexing by Pytorch
Calculating gradients by Pytorch
Simple classifier
Creating a Deep Network for MNIST by using PyTorch
vis.models
