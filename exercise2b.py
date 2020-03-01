import gym 
import numpy as np
env = gym.make('CartPole-v0')

pvariance = 0.1 # variance of initial parameters
ppvariance = 0.02 # variance of perturbations
nhiddens = 5 # number of hidden neurons
ninputs = env.observation_space.shape[0]
if (isinstance(env.action_space, gym.spaces.box.Box)):
    noutputs = env.action_space.shape[0]
else:
    noutputs = env.action_space.n 
    
# initialization of the training parameters 
def initialization_weights(nhiddens,ninputs,noutputs,pvariance):
    W1 = np.random.randn(nhiddens,ninputs) * pvariance # first layer
    W2 = np.random.randn(noutputs, nhiddens) * pvariance # second layer
    b1 = np.zeros(shape=(nhiddens, 1)) # bias first layer
    b2 = np.zeros(shape=(noutputs, 1)) # bias second layer
    return W1, W2, b1, b2

#update function of action
def update_activation(observation, W1, W2, b1, b2):
    observation.resize(ninputs,1)
    Z1 = np.dot(W1, observation) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = np.tanh(Z2)
    if (isinstance(env.action_space, gym.spaces.box.Box)):
        action = A2
    else:
        action = np.argmax(A2)

    return action

# update function of weights - training parameters
def update_weights(nhiddens,ninputs,noutputs, W1, W2, ppvariance):
    W1 += np.random.randn(nhiddens,ninputs) * ppvariance # first layer
    W2 += np.random.randn(noutputs, nhiddens) * ppvariance # second layer
    return W1, W2


#main function for neural network and cart pole environment execution
def population(weights, biases):
    for i in range(10):
        env.reset()
        env.render()
        action = env.action_space.sample()
        fitness = 0
        for _ in range(200):
            observation, reward, done, info = env.step(action)
            if reward == 1:
                fitness = fitness + 1
            action = update_activation(observation, weights[i,0], weights[i,1], biases[i,0], biases[i,1])
        weights[i,2] = fitness
    return weights


#initialization
weights = []
biases = []
for _ in range(10):
    W1, W2, b1, b2 = initialization_weights(nhiddens,ninputs,noutputs,pvariance)
    weights.append([W1, W2, 0])
    biases.append([b1,b2])
weights = np.asarray(weights)
biases = np.asarray(biases)
#main function execution
best_reward_on_iteration = 0
for k in range(50):
    weights = population(weights, biases)
    weights = weights[np.argsort(weights[:,2])]

    best_reward_on_iteration = weights[9,2]
    updated_weights = []
    
    #updating weights through tasks' algorithm
    for i in range(10):
        if i < 5:
            updated_weights.append([weights[i+5,0], weights[i+5,1], 0])
        else:
            W1, W2 = update_weights(nhiddens,ninputs,noutputs, weights[i,0], weights[i,1], ppvariance)
            updated_weights.append([W1, W2, 0])
    weights = np.asarray(updated_weights)

    print(k, best_reward_on_iteration)

    if best_reward_on_iteration == 200:
        break
env.close()
