import itertools
from trajectory_planner.trajectoryData import StateTrajectory
from trajectory_planner.environment import Environment
import tensorflow as tf
import numpy as np
from collections import deque
import random

class DeepQLearning:
    """
     Deep Q Learning Algorithm implementation
    """

    def __init__(self, environment, learning_rate = 0.00025, discount_factor = 0.4, N_hidden_layer = 2, layer_size = 32, eps_scheduler_rate = 0.1, batch_size = 32) -> None:
        """
        __init__ Initialize algorithm

        :param environment: Environment to learn
        :param learning_rate: learning rate of the optimizer, defaults to 0.001
        :type learning_rate: float, optional
        :param discount_factor: discount factor for future reward, defaults to 0.4
        :type discount_factor: float, optional
        :param N_hidden_layer: Number of hidden layer of the Q model, defaults to 2
        :type N_hidden_layer: int, optional
        :param layer_size: Size of each Dense layer of the Q model, defaults to 32
        :type layer_size: int, optional
        :param eps_scheduler_rate: decaying rate of the eps greedy scheduler, defaults to 0.1
        :type eps_scheduler_rate: float, optional
        """
        self.env = environment
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.N_hidden_layer = N_hidden_layer
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.eps_scheduler_rate = eps_scheduler_rate
        self.q_model = self.generate_q_model(self.N_hidden_layer, self.layer_size)
        self.q_target_model = self.generate_q_model(self.N_hidden_layer, self.layer_size)
        self.replay_memory = deque(maxlen=800_000)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
        # self.loss_function = tf.keras.losses.Huber()
        self.loss_function = tf.keras.losses.MeanSquaredError()

        self.episode_nr = 0
        self.accumulated_steps = 0

    def generate_q_model(self, N_hidden_layer=1, layer_size=24,
                         initializer=tf.keras.initializers.zeros()):
        """
        generate_q_model Generate the Neural Network approximating the Q function

        :param N_hidden_layer: Number of Hidden Layer, defaults to 1
        :type N_hidden_layer: int, optional
        :param layer_size: size of each hidden layer, defaults to 24
        :type layer_size: int, optional
        :return: Neural Network Model
        """
        model = tf.keras.Sequential()
        # Input
        model.add(tf.keras.layers.InputLayer(input_shape=(self.env.state_size,)))
        # Hidden Layer
        for i in range(N_hidden_layer):
            model.add(tf.keras.layers.Dense(
                layer_size, activation='relu', kernel_initializer=initializer))
        # Output
        model.add(tf.keras.layers.Dense(self.env.action_size,
                  activation='linear', kernel_initializer=initializer))
        return model

    def save_model(self, path: str):
        self.q_model.save(path)

    def load_model(self, path: str):
        self.q_model = tf.keras.models.load_model(path)
        self.q_target_model = tf.keras.models.load_model(path)

    def eps_greedy(self, eps, state):
        """
        eps_greedy epsilon greedy implementation

        :param eps: epsilon value
        :type eps: float, [0 ... 1]
        :param state: current state
        :return: index of the action
        :rtype: int
        """

        if np.random.rand() < eps:
            # Exploration
            return np.random.randint(0, self.env.action_size)

        # Exploitation
        q_values = self.q_model.predict(state.reshape((1, self.env.state_size)))
        greedy_action_index = np.argmax(q_values)
        return greedy_action_index

    def eps_scheduler(self):
        """
        eps_scheduler scheduler for calculating the epsilon depending on the episode_nr

        :return: epsilon value
        :rtype: float, [0 ... 1]
        """
        #return 1/(self.eps_scheduler_rate * self.episode_nr + 1) + 0.01
        initial_p = 1.0
        final_p = 0.001
        fraction = min(float(self.episode_nr) / self.eps_scheduler_rate, 1.0)
        return initial_p + fraction * (final_p - initial_p)

    def train(self, batch_size = 32):
        """
        train Train the Q model from the replay memory

        :param batch_size: Number of samples for the training, defaults to 32
        :type batch_size: int, optional
        """
        if len(self.replay_memory) < batch_size:
            batch_size = len(self.replay_memory)
        mini_batch = random.sample(self.replay_memory, batch_size)

        current_states = np.array([mem[0] for mem in mini_batch])
        next_states = np.array([mem[3] for mem in mini_batch])
        next_qs_list = self.q_target_model.predict(next_states)
        rewards = np.array([mem[2] for mem in mini_batch])
        actions = np.array([mem[1] for mem in mini_batch])
        dones = tf.convert_to_tensor([float(mem[4]) for mem in mini_batch])

        updated_q_values = rewards + self.discount_factor * tf.reduce_max(next_qs_list, axis=1) * (1 - dones)
        masks = tf.one_hot(actions, self.env.action_size)

        with tf.GradientTape() as tape:
            # Train the model on the states and updated Q-values
            q_values = self.q_model(current_states)

            # Apply the masks to the Q-values to get the Q-value for action taken
            q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
            # Calculate loss between new Q-value and old Q-value
            loss = self.loss_function(updated_q_values, q_action)

            # Backpropagation
            grads = tape.gradient(loss, self.q_model.trainable_variables)
            self.optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))





    def runEpisode(self, canvas_width, canvas_height):
        """
        runEpisode train for one episode

        :param canvas_width: width of the Canvas object
        :type canvas_width: int
        :param canvas_height: height of the Canvas object
        :type canvas_height: int
        :return: Trajectory from this episode, Sum of rewards of the episode
        """
        trajectory_data = StateTrajectory(canvas_width, canvas_height)
        total_training_rewards = 0
        current_state = self.env.reset()
        done = False
        ts = 0
        steps = 0

        while not done:
            action = self.eps_greedy(self.eps_scheduler(), current_state)
            next_step = self.env.step(action)
            trajectory_data.AppendState(current_state[:4], self.env.actions[action] ,ts)
            self.replay_memory.append(next_step)
            current_state = next_step[3]
            total_training_rewards += next_step[2]
            done = next_step[4]

            if not (self.accumulated_steps % 10):
                self.train(self.batch_size)

            if not(self.accumulated_steps % 200):
                self.q_target_model.set_weights(self.q_model.get_weights())

            ts += self.env.time_step
            steps += 1
            self.accumulated_steps += 1
        self.episode_nr += 1
        return trajectory_data, total_training_rewards
