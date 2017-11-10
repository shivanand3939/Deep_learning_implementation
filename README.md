# Deep_learning_implementation
Implement deep learning network from scratch 

Asks user for the following params :
1) Number of Hidden Layers
2) Width of each Hidden Layer
3) Maximum Number of iterations
4) Learning Rate
5) Regularization method, among Lasso and Ridge
6) Activation function for hidden layers, among 'tanh', 'sigmoid', 'relu'

Tips:
- Vary learning_rate absed on the loss. If loss is very high or Nan decrease the learning_rate (by factor of 10)
- Becomes a logistic regression if we choose 0 hidden layers 
