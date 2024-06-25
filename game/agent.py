from collections import deque
from game import SnakeGameAI, Direction, Block, BLOCK_SIZE
from model import create_model
import numpy as np
import tensorflow as tf
import random
from plot_game import plot, plot_stats

import os

MAX_MEMORY = 100_000
MAX_SCORE = 1000
BATCH_SIZE = 64
LR = 0.001

class Agent:

    def __init__(self, hidden_size=256,learning_rate=LR, gamma=0.9, epsilon_decay = 0.9584399559892547):
        self.n_games = 0
        self.epsilon = 1
        self.epsilon_min = 0.0001
        self.epsilon_decay = epsilon_decay
        self.hidden_size = hidden_size

        self.gamma = gamma
        self.memory = deque(maxlen=MAX_MEMORY)
        self.model = create_model(
            input_shape=[11], hidden_size=int(self.hidden_size), output_size=3)
        self.loss_fn = tf.keras.losses.MeanSquaredError()
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=LR)

    def epsilon_policy(self, state):
        # with probability epsilon select a random action (epsilon greedy policy)
        if self.epsilon > self.epsilon_min:
            # decrease epsilon by epsilon_decay
            self.epsilon = self.epsilon * self.epsilon_decay
        else:
            # decay stops when epsilon reaches epsilon_min
            self.epsilon = self.epsilon_min

        new_action = [0, 0, 0]

        # random action
        if np.random.rand() < self.epsilon:
            action_choice = np.random.randint(0, 2)
            new_action[action_choice] = 1
        # action based on Q-values
        else:
            Q_values = self.model.predict(state[np.newaxis])
            action_choice = np.argmax(Q_values[0])
            new_action[action_choice] = 1

        return new_action

    # sample random mini-batch of transitions from D
    def sample_experiences(self, batch_size):
        # select batch_size random indices from memory
        indices = np.random.randint(len(self.memory), size=batch_size)
        # extract the batch from memory
        batch = [self.memory[index] for index in indices]
        # extract states, actions, rewards, next_states, dones from batch
        states, actions, rewards, next_states, dones = [
            np.array([experience[field_index] for experience in batch])
            for field_index in range(5)
        ]
        return states, actions, rewards, next_states, dones

    def play_one_step(self, env, state):
        action = self.epsilon_policy(state)
        next_state, reward, done, info = env.play_step(action)
        self.memory.append((state, action, reward, next_state, done))
        return next_state, action, reward, done, info

    # set y_i with epsilon policy and perform a gradient descent
    def training_step(self, batch_size, return_loss = False):
        # sample a mini-batch of transitions from D
        experiences = self.sample_experiences(batch_size)
        states, actions, rewards, next_states, dones = experiences

        # target_Q_values = rewards + gamma * max_a' Q(s', a'), used for loss calculation
        target_Q_values = (rewards + self.gamma *
                           np.max(self.model.predict(next_states)))

        # Q_values = Q(s, a)
        next_Q_values = self.model.predict(next_states)
        max_next_Q_values = np.max(next_Q_values, axis=1)
        target_Q_values = (rewards + self.gamma * max_next_Q_values)

        with tf.GradientTape() as tape:
            all_Q_values = self.model(states)
            Q_values = tf.reduce_sum(
                all_Q_values * actions, axis=1, keepdims=True)
            loss = tf.reduce_mean(self.loss_fn(target_Q_values, Q_values))
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(
            zip(grads, self.model.trainable_variables))
        if return_loss:
            return loss.numpy()
    
    # function for Hyperparameter tuning with hyperopt
    def tune_hyper(self, agent):
        plot_scores = []
        plot_mean_scores = []
        total_score = 0
        reward_list = []
        reward_accumulator = 0
        record = 0
        agent = agent
        env = SnakeGameAI()
        env.reset()

        state = env._get_state()

        while agent.n_games <= 100:
            loss = 0
            next_state, action, reward, done, score = agent.play_one_step(
                env, state)

            state = next_state
            reward_accumulator += reward

            if done:
                agent.n_games += 1
                env.reset()
                plot_scores.append(score)
                reward_list.append(reward_accumulator)
                total_score += score
                mean_score = total_score / agent.n_games
                plot_mean_scores.append(mean_score)
                #plot(plot_scores, plot_mean_scores, rewards=reward_list, ngames=agent.n_games)

            if len(agent.memory) > BATCH_SIZE:
                loss = agent.training_step(BATCH_SIZE, return_loss = True)

        return loss       
        


if __name__ == "__main__":
    plot_scores = []
    plot_mean_scores = []
    loss_list = []
    reward_list = []
    reward_accumulator = 0
    loss_game_list = []
    total_score = 0
    record = 0
    agent = Agent()
    env = SnakeGameAI()
    env.reset()

    state = env._get_state()
    
    # episode loop
    while agent.n_games <=100:
        # time loop
        next_state, action, reward, done, score = agent.play_one_step(
            env, state)
        state = next_state
        reward_accumulator += reward

        if done:
            agent.n_games += 1
            env.reset()
            plot_scores.append(score)
            reward_list.append(reward_accumulator)
            total_score += score
            mean_score = total_score / agent.n_games
            plot_mean_scores.append(mean_score)
            plot(scores = plot_scores, mean_scores=plot_mean_scores,rewards = reward_list, ngames=agent.n_games)
            #loss_game_list.append(loss)
            reward_accumulator = 0

        if len(agent.memory) > BATCH_SIZE:
            loss = agent.training_step(BATCH_SIZE, return_loss = True)
            loss_list.append(loss)

    #plot_stats(loss_list)
            


            
