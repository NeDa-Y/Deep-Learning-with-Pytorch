# Deep-Learning-with-Pytorch

# PyTorch
PyTorch is another widely used framework for deep learning that has gained popularity due to its dynamic computational graph feature. Developed by Facebook's AI Research lab, PyTorch emphasizes ease of use and flexibility. It provides an intuitive interface that enables researchers to experiment with different network architectures and easily debug their code. PyTorch also offers a wide range of pre-trained models through its torchvision library, making it convenient for transfer learning tasks.
# Key Features:
•	Eager execution: PyTorch allows developers to define computational graphs dynamically during runtime, facilitating easy debugging and experimentation.
# •	Natural integration with Python: As a Python library, PyTorch seamlessly integrates with the Python ecosystem, enabling users to leverage popular libraries like NumPy and SciPy.
# •	Automatic differentiation: It provides automatic differentiation capabilities, which greatly simplifies the process of computing gradients for backpropagation.
# •	Mobile deployment support: PyTorch supports exporting models to mobile platforms using tools like TorchScript, allowing developers to deploy their models on smartphones and other edge devices.

# Trying To implement some of examples and exercises in Pytorch to review my knowledge
Tensors are the PyTorch equivalent to Numpy arrays. A vector is a 1-D tensor, and a matrix a 2-D tensor. When working with neural networks, we will use tensors of various shapes and number of dimensions.\
# torch.zeros: Creates a tensor filled with zeros\
# torch.ones: Creates a tensor filled with ones\
# torch.rand: Creates a tensor with random values uniformly sampled between 0 and 1\
# torch.randn: Creates a tensor with random values sampled from a normal distribution with mean 0 and variance 1\
# torch.arange: Creates a tensor containing the values N,N+1,N+2,...,M\
# torch.Tensor (input list): Creates a tensor from the list elements you provide\
obtain the shape of a tensor in the same way as in numpy (x.shape), or using the .size method.\
Converting Tensor to Numpy, and Numpy to Tensor\
To transform a numpy array into a tensor, we can use the function torch.from_numpy.\
To transform a PyTorch tensor back to a numpy array, we can use the function .numpy() on tensors.\
Operations by Pytorch\
Indexing by Pytorch\
Calculating gradients by Pytorch\
Simple classifier\
Creating a Deep Network for MNIST by using PyTorch\
vis.models



