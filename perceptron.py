import numpy as np
import random

'''
n_input = np.array([[100, 30], 
                    [150, 50], 
                    [90, 25], 
                    [175, 46]])
n_output = np.array([1, 0, 1, 0]).T #.T turns horizontal to vertical

print(n_input, n_output)
'''
class Perceptron(object):
    def __init__(self, w, b):
        self.w = w
        self.b = b
        
    def heaveside(self, n):
        if n > 0:
            return 1
        return 2
        
    def process(self, n):
        think = self.b
        for i in range(len(n)):
            think += n[i] * self.w[i]
        return self.heaveside(think)
        
    def adjust(self, n, delta, rate):
        for i in range(len(n)):
            self.w[i] += n * delta * rate
        self.b += delta * rate
        
    
def make_perceptron(n): #takes in matrix of inputs
    w = []
    for i in range(len(n)):
        w[i] = random.randint(-1, 1)
    return Perceptron(np.array(w), random.randint(-1, 1))
    
def main():
    pass