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
    
# initialization the training parameters 
def initialization_weights(nhiddens,ninputs,noutputs,pvariance):
    W1 = np.random.randn(nhiddens,ninputs) * pvariance 
    W2 = np.random.randn(noutputs, nhiddens) * pvariance 
    b1 = np.zeros(shape=(nhiddens, 1)) 
    b2 = np.zeros(shape=(noutputs, 1)) 
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
    W1 += np.random.randn(nhiddens,ninputs) * ppvariance 
    W2 += np.random.randn(noutputs, nhiddens) * ppvariance 
    return W1, W2

#main function for neural network and cart pole environment execution
def population(weights, biases, nhiddens,ninputs,noutputs, ppvariance):
    best_fitness = 0
    for _ in range(10):
        env.reset()
        env.render()
        action = env.action_space.sample()
        fitness = 0
        for _ in range(200):
            observation, reward, done, info = env.step(action)
            if reward == 1:
                fitness = fitness + 1
            action = update_activation(observation, weights[0,0], weights[0,1], biases[0,0], biases[0,1])
        if best_fitness < fitness:
            best_fitness = fitness
            best_w1 = weights[0,0]
            best_w2 = weights[0,1]
        weights[0,0], weights[0,1] = update_weights(nhiddens,ninputs,noutputs, best_w1, best_w2, ppvariance)
    weights[0,0]=best_w1
    weights[0,1] = best_w2
    weights[0,2] = best_fitness
    return weights



weights = []
biases = []
#initizlization
W1, W2, b1, b2 = initialization_weights(nhiddens,ninputs,noutputs,pvariance)
weights.append([W1, W2, 0])
biases.append([b1,b2])
weights = np.asarray(weights)
biases = np.asarray(biases)
best_reward_on_iteration = 0
#main function execution
weights = population(weights, biases,nhiddens,ninputs,noutputs, ppvariance)
#printing best weights and reward of neural network
print(weights)
env.close()
