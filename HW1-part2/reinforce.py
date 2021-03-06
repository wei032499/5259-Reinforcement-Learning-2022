# Spring 2022, IOC 5259 Reinforcement Learning
# HW1-partII: REINFORCE and baseline

import gym
from itertools import count
from collections import namedtuple
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
import torch.optim.lr_scheduler as Scheduler

# check and use GPU if available if not use CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define a useful tuple (optional)
SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])

        
class Policy(nn.Module):
    """
        Implement both policy network and the value network in one model
        - Note that here we let the actor and value networks share the first layer
        - Feel free to change the architecture (e.g. number of hidden layers and the width of each hidden layer) as you like
        - Feel free to add any member variables/functions whenever needed
        TODO:
            1. Initialize the network (including the shared layer(s), the action layer(s), and the value layer(s)
            2. Random weight initialization of each layer
    """
    def __init__(self):
        super(Policy, self).__init__()
        
        # Extract the dimensionality of state and action spaces
        self.discrete = isinstance(env.action_space, gym.spaces.Discrete)
        self.observation_dim = env.observation_space.shape[0]
        self.action_dim = env.action_space.n if self.discrete else env.action_space.shape[0]
        self.hidden_size = 128
        
        ########## YOUR CODE HERE (5~10 lines) ##########

        self.fc1 = nn.Linear(self.observation_dim, self.hidden_size)
        self.fc2_policy = nn.Linear(self.hidden_size, self.action_dim)
        self.fc2_value = nn.Linear(self.hidden_size, 1)

        ########## END OF YOUR CODE ##########
        
        # action & reward memory
        self.saved_actions = []
        self.rewards = []

    def forward(self, state):
        """
            Forward pass of both policy and value networks
            - The input is the state, and the outputs are the corresponding 
              action probability distirbution and the state value
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########

        x = F.relu(self.fc1(state))

        policy = self.fc2_policy(x)
        action_prob = F.softmax(policy, dim=0)

        state_value = self.fc2_value(x)

        ########## END OF YOUR CODE ##########

        return action_prob, state_value


    def select_action(self, state):
        """
            Select the action given the current state
            - The input is the state, and the output is the action to apply 
            (based on the learned stochastic policy)
            TODO:
                1. Implement the forward pass for both the action and the state value
        """
        
        ########## YOUR CODE HERE (3~5 lines) ##########
        
        action_prob, state_value = self.forward(torch.from_numpy(state).to(device))
        m = Categorical(action_prob)
        action = m.sample()

        ########## END OF YOUR CODE ##########
        
        # save to action buffer
        self.saved_actions.append(SavedAction(m.log_prob(action), state_value))

        return action.item()


    def calculate_loss(self, gamma=0.99):
        """
            Calculate the loss (= policy loss + value loss) to perform backprop later
            TODO:
                1. Calculate rewards-to-go required by REINFORCE with the help of self.rewards
                2. Calculate the policy loss using the policy gradient
                3. Calculate the value loss using either MSE loss or smooth L1 loss
        """
        
        # Initialize the lists and variables
        R = 0
        saved_actions = self.saved_actions
        policy_losses = [] 
        value_losses = [] 
        returns = []

        ########## YOUR CODE HERE (8-15 lines) ##########

        # calculate Gt of every state in the trajectory
        returns = np.zeros_like(self.rewards)
        timesteps = range(len(self.rewards))
        
        for t in reversed(timesteps):
            R = self.rewards[t] + gamma*R
            returns[t] = R
        returns = torch.tensor(returns, dtype=torch.float32).to(device)

        # calculate the loss of state value by MSE 
        loss_function = torch.nn.MSELoss()
        values = torch.cat([a.value for a in saved_actions])
        value_loss = loss_function(values, returns)
        
        # calculate the policy loss by policy gradient
        gamma_list = torch.tensor([ gamma**i for i in range(len(saved_actions))], dtype=torch.float32).to(device)
        neg_log_prob = torch.stack([-a.log_prob for a in saved_actions])
        policy_loss = torch.sum( gamma_list * returns * neg_log_prob)


        loss = value_loss + policy_loss

        self.clear_memory()


        ########## END OF YOUR CODE ##########
        
        return loss

    def clear_memory(self):
        # reset rewards and action buffer
        del self.rewards[:]
        del self.saved_actions[:]


def train(lr=0.01):
    '''
        Train the model using SGD (via backpropagation)
        TODO: In each episode, 
        1. run the policy till the end of the episode and keep the sampled trajectory
        2. update both the policy and the value network at the end of episode
    '''    
    
    # Instantiate the policy model and the optimizer
    model = Policy().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    # Learning rate scheduler (optional)
    scheduler = Scheduler.StepLR(optimizer, step_size=100, gamma=0.9)
    
    # EWMA reward for tracking the learning progress
    ewma_reward = 0
    
    # run inifinitely many episodes
    for i_episode in count(1):
        # reset environment and episode reward
        state = env.reset()
        ep_reward = 0
        t = 0
        # Uncomment the following line to use learning rate scheduler
        scheduler.step()
        
        # For each episode, only run 9999 steps so that we don't 
        # infinite loop while learning
        
        ########## YOUR CODE HERE (10-15 lines) ##########

        for _ in range(9999):
            t+=1
            action = model.select_action(state)
            state, reward, done, info = env.step(action)
            ep_reward += reward
            model.rewards.append(reward)

            if done:
                break
            
        
        loss = model.calculate_loss()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ########## END OF YOUR CODE ##########
            
        # update EWMA reward and log the results
        ewma_reward = 0.05 * ep_reward + (1 - 0.05) * ewma_reward
        print('Episode {}\tlength: {}\treward: {}\t ewma reward: {}'.format(i_episode, t, ep_reward, ewma_reward))

        # check if we have "solved" the cart pole problem
        if ewma_reward > env.spec.reward_threshold:
            torch.save(model.state_dict(), './preTrained/CartPole_{}.pth'.format(lr))
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(ewma_reward, t))
            break


def test(name, n_episodes=10):
    '''
        Test the learned model (no change needed)
    '''      
    model = Policy().to(device)
    
    model.load_state_dict(torch.load('./preTrained/{}'.format(name)))
    
    render = True
    max_episode_len = 10000
    
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        running_reward = 0
        for t in range(max_episode_len+1):
            action = model.select_action(state)
            state, reward, done, _ = env.step(action)
            running_reward += reward
            if render:
                 env.render()
            if done:
                break
        print('Episode {}\tReward: {}'.format(i_episode, running_reward))
    env.close()
    

if __name__ == '__main__':
    # For reproducibility, fix the random seed
    random_seed = 20  
    lr = 0.01
    env = gym.make('CartPole-v0')
    env.seed(random_seed)  
    torch.manual_seed(random_seed)  
    train(lr)
    test('CartPole_{}.pth'.format(lr))

