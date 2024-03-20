# Tiny NN 

In this project, I will build a tiny NN module to train on toy problems. This will build upon the Andrej Karpathy's `micrograd` NN. 

At its heart, this project is a way to impliment backpropagation for a NN. To do this we will create a tree stucture, so that gradient can be propergated back through the next.  

I have create a NN that can use a variety of activation function and take in different loss/error functions to train. It works well on simple cases, but its speed limits the complexity of the NN that can be created. As a result it doesn't work well on complicated data. It is also senstive to train params.


## Ref
- A. Karpathy's repo: https://github.com/karpathy/micrograd/tree/master. 