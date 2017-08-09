import numpy as np

# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T #4 by 1

'''
Weights represent synapse 1; X represents training input; y represents training output; 
'''

class NeuralNetwork:
    def __init__(self, X, y):
        self.X = X
        self. y = y
        self.generate_weights()
        
    def generate_weights(self):
        self.weights = 2 * np.random.random((self.X.shape[1], self.y.shape[1])) - 1 #columns of X by columns of y (3 x 1)
    
    def sigmoid(self, x, deriv = False):
        if(deriv==True):
            return x * (1 - x)
        return 1/(1 + np.exp(-x))
    
    def forward_propogate(self):
        self.y_hat = self.sigmoid(np.dot(self.X, self.weights))
        return self.y_hat
    
    def improve(self):
        #error (actual - predicted) * slope (derivative of sigmoid at prediction) * learning rate
        self.delta = (self.y - self.y_hat) * self.sigmoid(self.y_hat, True)
        self.weights += np.dot(X.T, self.delta)
        return self.weights

n = NeuralNetwork(X, y)

for iter in range(1000):
    n.forward_propogate()
    n.improve()
    print(n.y_hat - n.y)