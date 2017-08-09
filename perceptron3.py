import numpy as np

def sigmoid(x, deriv = False): #turn to between -1 and grabs derivative of sigmoid
    if not deriv:
        return x * (1 - x) #rate of change at x
    return 1/(1 + np.exp(-x))

n_input = np.array([[100, 30], 
                    [150, 50], 
                    [90, 25], 
                    [175, 46]])
                    
n_output = np.array([1, 0, 1, 0]).T #.T turns horizontal to vertical

syn0 = 2 * np.random.random((2, 4)) - 1 #makes 3 x 4 matrix (column x row), with random negative values, first weight
syn1 = 2 * np.random.random((4, 1)) - 1 #4 x 1 matrix, second weight

for i in xrange(10):
    L1 = sigmoid(np.dot(n_input, syn0)) #activation of the dot product of input and first weight
    L2 = sigmoid(np.dot(L1, syn1)) #activation of the dot product of layer 1 and second weight
    L2_delta = (n_output - L2) * (sigmoid(L2, True)) #derivative of L2 aka y hat (cost * sigmoid derivative of y hat), or error * slope
    L1_delta = L2_delta.dot(syn1.T) * (L1 * (1 - L1)) #L2 derivative (cost * sigmoid derivative) * 
    syn1 += L1.T.dot(L2_delta)
    syn0 += n_input.T.dot(L1_delta)