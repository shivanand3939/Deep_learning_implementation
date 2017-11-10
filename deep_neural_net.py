#BackPropagation
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)
Nclass = 500

def sigmoid(x):
    '''Compute Sigmoid of x'''
    return 1/(1+np.exp(-x))

def tanh(x):
    '''Compute tanh of x'''
    return (np.exp(x) - np.exp(-x))/(np.exp(x) + np.exp(-x))

def relu(x): 
    '''Compute relu of x'''
    return np.vectorize(lambda y: max(y,0.))(x).reshape(x.shape)

def my_data(depth, width_hidden): 
    '''Compute X, Y and initial Weight params
    @depth- number of hidden layers
    @width_hidden- number of neurons for each layer'''
    X1 = np.random.randn(Nclass, 2) + np.array([0, -2])
    X2 = np.random.randn(Nclass, 2) + np.array([2, 2])
    X3 = np.random.randn(Nclass, 2) + np.array([-2, 2])
    X = np.vstack((X1, X2, X3))
    y = np.array([0]*Nclass + [1]*Nclass + [2]*Nclass)
    plt.scatter(X[:,0], X[:,1], c=y, s = 100, alpha = 0.5)
    plt.show()
    # randomly initialize weights
    
    D = X.shape[1] # dimensionality of input
    K = 3 # number of classes
    M = 40 # hidden layer size
    
    weights = []
    for width in width_hidden:
        W = np.random.randn(D, width)
        b = np.random.randn(1, width)
        D = width
        weights.append([W, b])
        
    #Resetting D
    D = X.shape[1]  
    if width_hidden:
        v = np.random.randn(width_hidden[-1], K)
        c = np.random.randn(1, K)
    else: 
        v = np.random.randn(D, K)
        c = np.random.randn(1, K)
    weights.append([v, c])
    
    return X, y, weights

def nonlinear_activation(Y, func):
    '''Based on user input of nonlinear activation return the corresponding nonlinear function '''
    if func == 'sigmoid':
        return sigmoid(Y)
    elif func == 'tanh':
        return tanh(Y)
    elif func == 'relu':
        return relu(Y)
    else:
        raise 'select a valid non linear activation function' 

def first_derivative_Z(Y, func):
    '''Comupte the first derivative of a nonlinear activation function'''
    if func == 'sigmoid':
        return (Y * (1-Y))
    elif func == 'tanh':
        return (1 - Y*Y)
    elif func == 'relu':
        return np.vectorize(lambda x: 1 if x>0 else 0)(Y).reshape(Y.shape)
    else:
        raise 'select a valid non linear activation function' 
        
#FORWARD
def forward_prop(X, W, func = 'sigmoid'):
    '''compute forward propagation'''
    X = X[:]
    W = W[:]
    res = []
    res.append(X)
    for each in W[:-1]:
        weight = each[0]
        bias = each[1] 
        hidden = nonlinear_activation(X.dot(weight) + bias, func) 
        res.append(hidden)  
        X = hidden
    hidden = res[-1]
    sum_o = np.exp(hidden.dot(W[-1][0]) + W[-1][1]).sum(axis = 1, keepdims = True) 
    output = np.exp(hidden.dot(W[-1][0]) + W[-1][1])/sum_o  
    res.append(output)
    return res

def regularised_weights(W, regulariser, lamda):  
    '''compute delta_weights for chosen regularisation param'''
    if regulariser == 'R':
        return lamda*W
    elif regulariser == 'L':
        return lamda*np.vectorize(lambda y: 1 if y>0 else 0 if y==0 else -1)(W)
    else:
        return 0
    
#BACKPROP
def backward_prop(X, Y, W, forward_p, learning_rate, regulariser, lamda, func = 'sigmoid'):
    '''Compute backpropagation'''
    dJdW = {}  
    for i in range(len(W)):
        if i == 0:
            dJdW[i] = Y - forward_p[-1]  
            W[-1][0] += learning_rate*((forward_p[-2].T).dot(dJdW[0]) - regularised_weights(W[-1][0],regulariser, lamda)) 
            W[-1][1] += learning_rate*((dJdW[0]).sum(axis = 0) - regularised_weights(W[-1][1],regulariser, lamda))
        else:     
            val = dJdW[i-1].dot(W[-i][0].T)*first_derivative_Z(forward_p[-i-1], func)  
            #There is no connection from one layer to the next layer's Bias unit. Hence need not compute val_bias
            dJdW[i] = val
            W[-i-1][0] += learning_rate*((forward_p[-i-2].T).dot(dJdW[i]) - regularised_weights(W[-i-1][0],regulariser, lamda))
            W[-i-1][1] += learning_rate*((dJdW[i]).sum(axis = 0) - regularised_weights(W[-i-1][1],regulariser, lamda))
    return W

#TRAIN MODEL
def train_model(X, Y, W, max_iter, learning_rate, regulariser, lamda, func):
    '''perform training'''
    Y_reshape = np.zeros((len(Y), len(np.unique(Y))))
    forward_prop_res = forward_prop(X, W, func)
    for k, v in enumerate(Y):
        Y_reshape[k, v] = 1  
    iters = 0
    losses = []
    while  iters < max_iter:
        W = backward_prop(X, Y_reshape, W,forward_prop_res, learning_rate, regulariser, lamda, func)
        forward_prop_res = forward_prop(X, W, func) 
        loss = -(np.log(forward_prop_res[-1])*(Y_reshape)).sum() + (0.5*lamda*sum([(wt[0]*wt[0]).sum()+(wt[1]*wt[1]).sum() for wt in W]) if regulariser == 'R' \
               else lamda*sum([ np.abs(wt[0]).sum()+np.abs(wt[1]).sum() for wt in W]) if regulariser == 'L' else 0) 
        if iters % 100 == 0:
            print 'Training accuracy is : ', classification_accuracy(Y, forward_prop_res[-1].argmax(axis=1))
            print 'Training loss is : ', loss  
            losses.append(loss)
        iters += 1 
    plt.plot(range(len(losses)), losses)
    plt.show()
    return W

#CLASSIFICATION
def classification_accuracy(Y, P):
    '''computes classification accuracy'''
    n_correct = 0
    n_total = 0
    for i in range(len(Y)):
        n_total += 1
        if Y[i] == P[i]:
            n_correct += 1
    return float(n_correct) / n_total

def train_test_split(X, Y):
    '''split our dataset into 80:20 '''
    from sklearn.utils import shuffle
    X, Y = shuffle(X,Y)
    return X[:-100], Y[:-100], X[-100:], Y[-100:]

if __name__ == '__main__':
    #Ask user all necessary inputs
    depth = int(raw_input("How many Hidden layers do you want? [Layers counted from input layer] "))
    width_hidden = []
    for h in range(depth):
        width_hidden.append(int(raw_input("How many neurons do you want for layer " + str(h) + ' -')))
    func = raw_input("Choose activation function for hidden layers 'tanh', 'sigmoid' or 'relu' ")
    if func not in ['tanh', 'sigmoid', 'relu']:
        func = 'sigmoid'
    max_iter = int(raw_input('Enter max iterations ')) 
    learning_rate = float(raw_input('Enter the learning rate ')) 
    is_regulariation = raw_input('Do you want a regularsiation "Y"/"N" ')
    regulariser = 0
    lamda = 0
    if is_regulariation.lower() == 'y':
        regulariser = raw_input('Choose between Ridge and Lasso regulariser - "R"/"L" ')
        lamda = float(raw_input('Great! so you want regularisation, what lamda do you prefer '))
    
    X, Y, W = my_data(depth, width_hidden)
    X_train, Y_train, X_test, Y_test = train_test_split(X, Y)
    W = train_model(X_train, Y_train, W, max_iter, learning_rate, regulariser, lamda, func) 
    test_output = forward_prop(X_test, W, func)[-1].argmax(axis=1)
    print 'Test Set Accuracy is : ', classification_accuracy(Y_test, test_output)
