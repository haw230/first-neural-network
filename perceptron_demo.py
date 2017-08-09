import numpy as np

# sigmoid function
def nonlin(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))
    
# input dataset
X = np.array([  [0,0,1],
                [0,1,1],
                [1,0,1],
                [1,1,1] ])
    
# output dataset            
y = np.array([[0,0,1,1]]).T #4 by 1

# seed random numbers to make calculation
# deterministic (just a good practice)
np.random.seed(1)

# initialize weights randomly with mean 0
syn0 = 2*np.random.random((3,1)) - 1 #3 rows by 1 column

for iter in xrange(10):

    # forward propagation
    l0 = X
    l1 = nonlin(np.dot(l0,syn0)) #4 by 1

    # how much did we miss?
    l1_error = y - l1 #4 by 1

    # multiply how much we missed by the 
    # slope of the sigmoid at the values in l1
    l1_delta = l1_error * nonlin(l1, True) #multiply error b slope of sigmoid
    print(l1_delta)
    print(l0.T)
    print(syn0)
    syn0 += np.dot(l0.T, l1_delta)
    '''
     l0.T (transposed output 3x4) * l1_delta (error * deriv 4x1) = matrix to add to syn0 (3x1)
    [[0 0 1 1]                  [[-0.00107879]              [[0 + 0 + 0.00047318 + 0.00071816]
     [0 1 0 1]                   [-0.00071341]               [0 + -0.00071341 + 0 + 0.00071816]
     [1 1 1 1]]                  [ 0.00047318]               [-0.00107879 + -0.00071341 + 0.00047318 + 0.00071816]
                                 [ 0.00071816]]              ]
                                 
    Transposed because those are the numbers that are affected by the weight in the first place
    
     syn0 (simultaneously updated)
    [[random weight 1]
     [random weight 2]
     [random weight 3]]
    '''

    '''
    Larger error (straight up wrong) or larger slope (not confident) will mean 
    greater change to weights. Sigmoids have higher rates of change closer to the middle and lower to none at -1 or 1 (S shape). So, that means the closer
    y hat (L2) is to -1 or 1, the more confident it is which means there is less that needs to be changed (it's confident about
    the estimate). The more drastic the slope, the less sure and the more that will be changed.
    '''
    # update weights

print "Output After Training:"
print l1
