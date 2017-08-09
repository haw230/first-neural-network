import numpy as np
import random



n_input = np.array([[100, 30], 
                    [150, 50], 
                    [90, 25], 
                    [175, 46]])
                    
n_output = np.array([1, 0, 1, 0]).T #.T turns horizontal to vertical

class Perceptron(object):
    def __init__(self, w):
        np.random.seed(1)
        self.w = w
        
    def heaveside(self, n):
        if n > 0:
            return 1
        return 2
        
    def process(self, n):
        think = 0
        for i in range(len(n)):
            think += n[i] * self.w[i]
        return self.heaveside(think)
        
    def adjust(self, n, delta, rate):
        for i in range(len(n)):
            self.w[i] += n * delta * rate
        
    def sigmoid(self, x, deriv = False):
        if not deriv:
            return x * (1 - x)
        return 1/(1 + np.exp(-x))
        

def make_perceptron(n): #takes in matrix of inputs
    w = []
    for i in range(len(n)):
        w[i] = random.randint(-1, 1)
    return Perceptron(np.array(w))
    
def main():
    pass

print(n_input, n_output)