import numpy as np
lis = []
dic = {}

#sigmoid function
def sigmoid(x, deriv=False):
    if deriv == True:
        return x*(1-x)
    return 1/(1+np.exp(-x))

#input
x = np.array([[1,0.9,0.1,0.8,0.85,0,0.7,0.8,1],
                      [0.9,0.9,0,1,0.6,0,0.3,0.5,0.7],
                      [0,0.2,1,0,0,0,0,1,0.2],
                      [0.3,0.3,1,0.65,0.7,0.7,0.6,0.2,0.5],
                      [0.35,0.4,0,0.1,0.4,0,0.75,0,0.3], 
                      [0,0,0.1,0.1,0.9,0.9,0.9,0,0.45],
                      [0,0,0.8,0,1,1,1,0,1]])
#intended output
y = np.array([[1,0,0,0,0,0,0],
                      [0,1,0,0,0,0,0],
                      [0,0,1,0,0,0,0],
                      [0,0,0,1,0,0,0],
                      [0,0,0,0,1,0,0],
                      [0,0,0,0,0,1,0],
                      [0,0,0,0,0,0,1]])

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

#main program
print('||Biome Guessing Neural Network||\nEnter the probability of each factor as a decimal (0 - 1)')
trees = float(input('trees: '))
shrubs = float(input('shrubs: '))
sand = float(input('sand: '))
grass = float(input('grass: '))
water = float(input('water: '))
ocean = float(input('ocean: '))
wet = float(input('wet: '))
hot = float(input('hot: '))
animals = float(input('animals: '))

l0 = np.array([[trees, shrubs, sand, grass, water, ocean, wet, hot, animals]])
l1 = sigmoid(np.dot(l0,syn0))
l2 = sigmoid(np.dot(l1,syn1))

for index, i in enumerate(range(7)):
    lis.append(l2[0][index])

dic['rainforest'] = round(lis[0], 3)
dic['boreal forest'] = round(lis[1], 3)
dic['desert'] = round(lis[2], 3)
dic['coastal'] = round(lis[3], 3)
dic['tundra'] = round(lis[4], 3)
dic['artic'] = round(lis[5], 3)
dic['marine'] = round(lis[6], 3)

#creating dictionaries of results
dic = sorted(dic.items(), key=lambda x:x[1], reverse=True)
dic = dict(dic)
dic3 = dict(list(dic.items())[0: 3])

#printing results
print('\nresults (3 most likely):')
for key, value in dic3.items():
    print(key + ':', value)
