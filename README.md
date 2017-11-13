# Deep_learning_implementation
Implement deep learning network from scratch 

***********************************************************************************
Example Usage:

clf = DLClassifier(max_iters, depth, width_hidden, learning_rate , activation_func = func, \
                 regulariser = regulariser, lamda = lamda)

#Using Demo data, but we can use any data 
X, Y = clf.my_data()

X_train, Y_train, X_test, Y_test = clf.train_test_split(X, Y, 0.2)

clf.fit(X_train, Y_train) 

test_output = clf.predict(X_test)

print 'Test Set Accuracy is : ', clf.classification_accuracy(Y_test, test_output)

***********************************************************************************
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
