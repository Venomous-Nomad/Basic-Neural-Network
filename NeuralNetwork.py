import numpy as np

#sigmoid function
def sigmoid(x, deriv=False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input
x = np.array([[0,0,1],
                      [1,1,1],
                      [1,0,1],
                      [0,1,1]])
#intended output
y = np.array([[0,1,1,0]]).T

#randomize the numbers being produced
np.random.seed(1)

#create the synapse layer as an array
syn0 = 2*np.random.random((3,1)) - 1

#training loop
for i in range(1000):
    #l0 is the input layer and is assigned the inputs 
    l0 = x
    #l1 is the output layer which is the sigmoid function of l0 and the synapse
    l1 = sigmoid(np.dot(l0,syn0))
    #error is the intended output subtracted by the actual output
    l1_error = y - l1

    #l1 delta is the previous error times by the sigmoid of the output layer
    l1_delta = l1_error * sigmoid(l1,True)
    #adjust the synapse using l0 and l1 delta
    syn0 += np.dot(l0.T,l1_delta)

#output
print('output after training:')
print(l1)
print('real output:')
print(y)
        
