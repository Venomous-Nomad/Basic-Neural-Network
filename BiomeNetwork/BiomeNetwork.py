import numpy as np

#sigmoid function
def sigmoid(x, deriv=False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input
x = np.array([[1,1,0,1,1,0,0.6,0.8,1],
                      [0,1,1,0,0,0,0,1,0]])
#intended output
y = np.array([[1,0,0,0,0,0,0],[0,0,1,0,0,0,0]])

#randomize the numbers being produced
np.random.seed(1)

#create the synapse layer as an array
syn0 = 2*np.random.random((9,9)) - 1
syn1 = 2*np.random.random((9,7)) - 1

#training loop
for i in range(1000):
    #l0 is the input layer and is assigned the inputs 
    l0 = x
    #l1 is the output layer which is the sigmoid function of l0 and the synapse
    l1 = sigmoid(np.dot(l0,syn0))
    l2 = sigmoid(np.dot(l1,syn1))
    #error is the intended output subtracted by the actual output
    l2_error = y - l2

    if i% 10000 == 0:
        print('Error:' + str(np.mean(np.abs(l2_error))))

    #l1 delta is the previous error times by the sigmoid of the output layer
    l2_delta = l2_error*sigmoid(l2,True)

    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid(l1,True)
    #adjust the synapse using l0 and l1 delta
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)

#output
print('output after training:')
print(l2)
print('real output:')
print(y)
        
