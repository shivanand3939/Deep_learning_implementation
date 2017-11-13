#BackPropagation
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

class DLClassifier():
    Nclass = 500
    weights = None
    
    #params 
    def __init__(self, max_iters = 1000, depth = 1, width_hidden = [10], learning_rate = 0.001, activation_func = 'tanh', \
                 regulariser = 'R', lamda =0.1):
        '''max_iters: Max iterations to be run on training data
           depth: Number of Hidden Layers
           width_hidden: Number of neurons in each hidden layer. Expecting a List
           learning_rate:
           activation_func = ['sigmoid', 'tanh', 'relu']
           regulariser = ['R', 'L'] select any other value for No regularisation
           lamda: the learning_rate parameter for Regulariser'''
    
        self.max_iters = max_iters
        self.depth = depth
        self.width_hidden = width_hidden
        self.learning_rate = learning_rate
        self.activation_func = activation_func
        self.regulariser = regulariser
        self.lamda = lamda
        
    def train_test_split(self, X, Y, percent_test_data = 0.2):
        '''split our dataset into 80:20 '''
        from sklearn.utils import shuffle
        len_test = int(len(X)*percent_test_data)
        X, Y = shuffle(X,Y)
        return X[:-len_test], Y[:-len_test], X[-len_test:], Y[-len_test:]
    
    def my_data(self): 
        '''Compute X, Y and initial Weight params
        @depth- number of hidden layers
        @width_hidden- number of neurons for each layer'''
        X1 = np.random.randn(self.Nclass, 2) + np.array([0, -2])
        X2 = np.random.randn(self.Nclass, 2) + np.array([2, 2])
        X3 = np.random.randn(self.Nclass, 2) + np.array([-2, 2])
        X = np.vstack((X1, X2, X3))
        y = np.array([0]*self.Nclass + [1]*self.Nclass + [2]*self.Nclass)
        plt.scatter(X[:,0], X[:,1], c=y, s = 100, alpha = 0.5)
        plt.show()
        self.get_initial_weights(X, y)
        return X, y
    
    def get_initial_weights(self, X, Y):
        D = X.shape[1] # dimensionality of input
        K = len(np.unique(Y)) # number of classes 
        weights = []
        # randomly initialize weights
        for width in self.width_hidden:
            W = np.random.randn(D, width)
            b = np.random.randn(1, width)
            D = width
            weights.append([W, b])
        #Resetting D
        D = X.shape[1]  
        if self.width_hidden:
            v = np.random.randn(self.width_hidden[-1], K)
            c = np.random.randn(1, K)
        else: 
            v = np.random.randn(D, K)
            c = np.random.randn(1, K)
        weights.append([v, c])
        self.weights = weights
    
    def sigmoid(self, x):
        '''Compute Sigmoid of x'''
        return 1/(1+np.exp(-x))

    def tanh(self, x):
        '''Compute tanh of x'''
        return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

    def relu(self, x): 
        '''Compute relu of x'''
        return np.vectorize(lambda y: max(y,0.))(x).reshape(x.shape)

    def nonlinear_activation(self, Y, func):
        '''Based on user input of nonlinear activation return the corresponding nonlinear function '''
        if func == 'sigmoid':
            return self.sigmoid(Y)
        elif func == 'tanh':
            return self.tanh(Y)
        elif func == 'relu':
            return self.relu(Y)
        else:
            raise 'select a valid non linear activation function' 
    
    #FORWARD
    def forward_prop(self, X, W, func = 'sigmoid'):
        '''compute forward propagation'''
        X = X[:]
        W = W[:]
        res = []
        res.append(X)
        for each in W[:-1]:
            weight = each[0]
            bias = each[1] 
            hidden = self.nonlinear_activation(X.dot(weight) + bias, func) 
            res.append(hidden)  
            X = hidden
        hidden = res[-1]
        sum_o = np.exp(hidden.dot(W[-1][0]) + W[-1][1]).sum(axis = 1, keepdims = True) 
        output = np.exp(hidden.dot(W[-1][0]) + W[-1][1])/sum_o  
        res.append(output)
        return res

    def first_derivative_Z(self, Y, func):
        '''Comupte the first derivative of a nonlinear activation function'''
        if func == 'sigmoid':
            return (Y * (1-Y))
        elif func == 'tanh':
            return (1 - Y*Y)
        elif func == 'relu':
            return np.vectorize(lambda x: 1 if x>0 else 0)(Y).reshape(Y.shape)
        else:
            raise 'select a valid non linear activation function' 

    def regularised_weights(self, W, regulariser, lamda):  
        '''compute delta_weights for chosen regularisation param'''
        if regulariser == 'R':
            return lamda*W
        elif regulariser == 'L':
            return lamda*np.vectorize(lambda y: 1 if y>0 else 0 if y==0 else -1)(W)
        else:
            return 0

    #BACKPROP
    def backward_prop(self, X, Y, W, forward_p, learning_rate, regulariser, lamda, func = 'sigmoid'):
        '''Compute backpropagation'''
        dJdW = {}  
        for i in range(len(W)):
            if i == 0:
                dJdW[i] = Y - forward_p[-1]  
                W[-1][0] += learning_rate*((forward_p[-2].T).dot(dJdW[0]) - self.regularised_weights(W[-1][0],regulariser, lamda)) 
                W[-1][1] += learning_rate*((dJdW[0]).sum(axis = 0) - self.regularised_weights(W[-1][1],regulariser, lamda))
            else:     
                val = dJdW[i-1].dot(W[-i][0].T)*self.first_derivative_Z(forward_p[-i-1], func)  
                #There is no connection from one layer to the next layer's Bias unit. Hence need not compute val_bias
                dJdW[i] = val
                W[-i-1][0] += learning_rate*((forward_p[-i-2].T).dot(dJdW[i]) - self.regularised_weights(W[-i-1][0],regulariser, lamda))
                W[-i-1][1] += learning_rate*((dJdW[i]).sum(axis = 0) - self.regularised_weights(W[-i-1][1],regulariser, lamda))
        return W
    
    #USER FACING FUNCTIONS are defined below     
    def fit(self, X, Y):
        '''perform training'''
        if self.weights is None:
            W = self.get_initial_weights(X,Y)
        W = self.weights
        
        Y_reshape = np.zeros((len(Y), len(np.unique(Y))))
        forward_prop_res = self.forward_prop(X, W, self.activation_func)
        for k, v in enumerate(Y):
            Y_reshape[k, int(v)] = 1  
        iters = 0
        losses = []
        while  iters < self.max_iters:
            W = self.backward_prop(X, Y_reshape, W,forward_prop_res, self.learning_rate, self.regulariser, self.lamda, self.activation_func)
            forward_prop_res = self.forward_prop(X, W, self.activation_func) 
            loss = -(np.log(forward_prop_res[-1])*(Y_reshape)).sum() + (0.5*self.lamda*sum([(wt[0]*wt[0]).sum()+(wt[1]*wt[1]).sum() for wt in W]) if regulariser == 'R' \
                   else self.lamda*sum([ np.abs(wt[0]).sum()+np.abs(wt[1]).sum() for wt in W]) if regulariser == 'L' else 0) 
            if iters % 100 == 0:
                print 'Training accuracy is : ', self.classification_accuracy(Y, forward_prop_res[-1].argmax(axis=1))
                print 'Training loss is : ', loss  
                losses.append(loss)
            iters += 1 
        plt.plot(range(len(losses)), losses)
        plt.show()
        self.weights = W

    def predict(self, X):
        return self.forward_prop(X, self.weights, self.activation_func)[-1].argmax(axis=1)
    
    def classification_accuracy(self, Y, P):
        '''computes classification accuracy'''
        n_correct = 0
        n_total = 0
        for i in range(len(Y)):
            n_total += 1
            if Y[i] == P[i]:
                n_correct += 1
        return float(n_correct) / n_total


if __name__ == '__main__':
    #Ask user all necessary inputs
    depth = int(raw_input("How many Hidden layers do you want? [Layers counted from input layer] "))
    width_hidden = []
    for h in range(depth):
        width_hidden.append(int(raw_input("How many neurons do you want for layer " + str(h) + ' -')))
    func = raw_input("Choose activation function for hidden layers 'tanh', 'sigmoid' or 'relu' ")
    if func not in ['tanh', 'sigmoid', 'relu']:
        func = 'sigmoid'
    max_iters = int(raw_input('Enter max iterations ')) 
    learning_rate = float(raw_input('Enter the learning rate ')) 
    is_regulariation = raw_input('Do you want a regularsiation "Y"/"N" ')
    regulariser = 0
    lamda = 0
    if is_regulariation.lower() == 'y':
        regulariser = raw_input('Choose between Ridge and Lasso regulariser - "R"/"L" ')
        lamda = float(raw_input('Great! so you want regularisation, what lamda do you prefer '))
        
    clf = DLClassifier(max_iters, depth, width_hidden, learning_rate , activation_func = func, \
                 regulariser = regulariser, lamda = lamda)
    #Using Demo data, but we can use any data 
    X, Y = clf.my_data()
    X_train, Y_train, X_test, Y_test = clf.train_test_split(X, Y, 0.2)
    clf.fit(X_train, Y_train) 
    test_output = clf.predict(X_test)
    print 'Test Set Accuracy is : ', clf.classification_accuracy(Y_test, test_output)
