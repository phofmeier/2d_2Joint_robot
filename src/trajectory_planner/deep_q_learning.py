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

        self.replay_memory = deque(maxlen=800_000)
        # schedule learning rate 
        learning_rate_scheduler = tf.keras.optimizers.schedules.PolynomialDecay(0.01, 30000, end_learning_rate=0.0001, power=1.0, cycle=False, name=None)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_scheduler, clipnorm=10.0)
        self.loss_function = tf.keras.losses.Huber()
        self.loss_function = tf.keras.losses.MeanSquaredError()
        self.normalizer = tf.keras.layers.Normalization()
        #self.loss_function = tf.keras.losses.MeanSquaredError()
        self.q_model = self.generate_q_model(self.N_hidden_layer, self.layer_size)
        self.q_target_model = self.generate_q_model(self.N_hidden_layer, self.layer_size)
        self.episode_nr = 0
        self.accumulated_steps = 0
        self.used_action_episode = [0] * self.env.action_size
        self.used_action_accumulated = [0] * self.env.action_size
        self.current_loss = 0
        self.trainings = 0
        self.episode_q_min = 0
        self.episode_q_max = 0
        self.episode_q_mean = 0

    def learning_rate_schedule(self):
        initial_p = 1.0
        final_p = 0.0001
        scheduler_rate = 700
        fraction = min(float(self.episode_nr) / scheduler_rate, 1.0)
        return initial_p + fraction * (final_p - initial_p)


    def generate_q_model(self, N_hidden_layer=1, layer_size=24,
                         initializer=tf.keras.initializers.RandomUniform(-0.01, 0.01)):
        """
        generate_q_model Generate the Neural Network approximating the Q function

        :param N_hidden_layer: Number of Hidden Layer, defaults to 1
        :type N_hidden_layer: int, optional
        :param layer_size: size of each hidden layer, defaults to 24
        :type layer_size: int, optional
        :return: Neural Network Model
        """
        #initializer = tf.keras.initializers.HeNormal()
        initializer = tf.keras.initializers.GlorotUniform()
        initializer_bias = tf.keras.initializers.RandomUniform(minval=0.0, maxval=0.001)
        initializer_bias_output = tf.keras.initializers.RandomUniform(minval=-0.1, maxval=0.1)
        #initializer_output = tf.keras.initializers.RandomNormal(mean=0.0, stddev=0.05)
        initializer_output = tf.keras.initializers.GlorotNormal()


        model = tf.keras.Sequential()
        # Input
        model.add(tf.keras.layers.InputLayer(input_shape=(self.env.state_size,)))
        model.add(self.normalizer)
        # Hidden Layer
        for i in range(N_hidden_layer):
            model.add(tf.keras.layers.Dense(
                layer_size, activation='relu', kernel_initializer=initializer, bias_initializer=initializer_bias))
        # Output
        model.add(tf.keras.layers.Dense(self.env.action_size,
                  activation='sigmoid', kernel_initializer=initializer_output, bias_initializer=initializer_bias_output))
        
        model.compile(optimizer = self.optimizer, loss = self.loss_function, metrics=[tf.keras.metrics.MeanAbsoluteError()])
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
        greedy_action_index = np.argmin(q_values)
        # print(q_values, greedy_action_index)
        self.used_action_episode[greedy_action_index] += 1
        self.used_action_accumulated[greedy_action_index] += 1
        return greedy_action_index

    def generate_random_data(self, nr_episodes):
        

        for i in range(nr_episodes):
            current_state = self.env.reset()
            done = False
            while not done:
                action = self.eps_greedy(1.0, current_state)
                next_step = self.env.step(action)
                self.replay_memory.append(next_step)
                current_state = next_step[3]
                done = next_step[4]
        

    def fit_normalizer(self):
        current_states = np.array([mem[0] for mem in self.replay_memory])
        self.normalizer.adapt(current_states)
        self.q_model.compile(optimizer = self.optimizer, loss = self.loss_function, metrics=[tf.keras.metrics.MeanAbsoluteError()])
        self.q_target_model.compile(optimizer = self.optimizer, loss = self.loss_function, metrics=[tf.keras.metrics.MeanAbsoluteError()])



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
            return
            # batch_size = len(self.replay_memory)
        self.trainings += 1
        mini_batch = random.sample(self.replay_memory, batch_size)

        current_states = np.array([mem[0] for mem in mini_batch])
        next_states = np.array([mem[3] for mem in mini_batch])
        next_qs_list = self.q_target_model.predict(next_states)
        rewards = np.array([mem[2] for mem in mini_batch])
        actions = np.array([mem[1] for mem in mini_batch])
        dones = tf.convert_to_tensor([float(mem[4]) for mem in mini_batch])
        q_values = self.q_model.predict(current_states)
        #print("states", current_states)
        updated_q_values = rewards + self.discount_factor * tf.reduce_min(next_qs_list, axis=1) * (1 - dones)
        #print("Train: updated_q_values", updated_q_values, "rewards", rewards, "q_values", q_values, "next_qs", next_qs_list)
        masks = tf.one_hot(actions, self.env.action_size)
        
        y = tf.multiply(1-masks,q_values) + tf.multiply(masks,tf.reshape(updated_q_values, [batch_size, 1]))
        #print(y)
        metrices = self.q_model.fit(current_states, y, batch_size=batch_size, verbose=0)  
        self.current_loss = self.current_loss + 1.0/self.trainings * (metrices.history["loss"][0] - self.current_loss)
        self.episode_q_min += 1.0/self.trainings * (np.min(y)- self.episode_q_min)
        self.episode_q_max += 1.0/self.trainings * (np.max(y)- self.episode_q_max)
        self.episode_q_mean += 1.0/self.trainings * (np.mean(y)- self.episode_q_mean)
        # with tf.GradientTape() as tape:
        #     # Train the model on the states and updated Q-values
            

        #     # Apply the masks to the Q-values to get the Q-value for action taken
        #     q_action = tf.reduce_sum(tf.multiply(q_values, masks), axis=1)
        #     # Calculate loss between new Q-value and old Q-value
        #     loss = self.loss_function(updated_q_values, q_action)

        #     # Backpropagation
        #     grads = tape.gradient(loss, self.q_model.trainable_variables)
        #     self.optimizer.apply_gradients(zip(grads, self.q_model.trainable_variables))





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
        self.used_action_episode = [0] * self.env.action_size
        self.trainings = 0
        self.current_loss = 0
        self.episode_q_min = 0
        self.episode_q_max = 0
        self.episode_q_mean = 0

        while not done:
            action = self.eps_greedy(self.eps_scheduler(), current_state)
            next_step = self.env.step(action)
            trajectory_data.AppendState(current_state[:4], self.env.actions[action] ,ts)
            self.replay_memory.append(next_step)
            current_state = next_step[3]
            total_training_rewards += next_step[2]
            done = next_step[4]

            if not (self.accumulated_steps % 4):
                self.train(self.batch_size)

            if not(self.accumulated_steps % 40):
                self.q_target_model.set_weights(self.q_model.get_weights())

            ts += self.env.time_step
            steps += 1
            self.accumulated_steps += 1
        self.episode_nr += 1
        print("Episode:", self.episode_nr, "Action Epsiode:", self.used_action_episode, "Action Accumulated:", self.used_action_accumulated)
        print("Loss:", self.current_loss)
        print("Q Values min:", self.episode_q_min, "max:", self.episode_q_max, "mean:", self.episode_q_mean)
        return trajectory_data, total_training_rewards
