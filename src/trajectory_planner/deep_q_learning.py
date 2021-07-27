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

    def __init__(self, environment, learning_rate = 0.001, discount_factor = 0.4, N_hidden_layer = 2, layer_size = 32, eps_scheduler_rate = 0.1) -> None:
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
        self.eps_scheduler_rate = eps_scheduler_rate
        self.q_model = self.generate_q_model(self.N_hidden_layer, self.layer_size)
        self.replay_memory = deque(maxlen=50_000)

        self.episode_nr = 0

    def generate_q_model(self, N_hidden_layer=1, layer_size=24):
        """
        generate_q_model Generate the Neural Network approximating the Q function

        :param N_hidden_layer: Number of Hidden Layer, defaults to 1
        :type N_hidden_layer: int, optional
        :param layer_size: size of each hidden layer, defaults to 24
        :type layer_size: int, optional
        :return: Neural Network Model
        """
        init = tf.keras.initializers.HeUniform()
        model = tf.keras.Sequential()
        # Input
        model.add(tf.keras.layers.InputLayer(input_shape=(self.env.state_size,)))
        # Hidden Layer
        for i in range(N_hidden_layer):
            model.add(tf.keras.layers.Dense(layer_size, activation='relu', kernel_initializer=init))
        # Output
        model.add(tf.keras.layers.Dense(self.env.action_size,
                  activation='linear', kernel_initializer=init))
        model.compile(loss=tf.keras.losses.Huber(), optimizer=tf.keras.optimizers.Adam(
        learning_rate=self.learning_rate), metrics=['accuracy'])
        return model

    def eps_greedy(self, eps, state):
        """
        eps_greedy epsilon greedy implementation

        :param eps: epsilon value
        :type eps: float, [0 ... 1]
        :param state: current state
        :return: index of the action
        :rtype: int
        """
        random_number = np.random.rand()
        if random_number < eps:
            # Exploration
            return np.random.randint(0, self.env.action_size)

        # Exploitation
        q_values = self.q_model.predict(state.reshape((1, self.env.state_size)))
        print(q_values)
        return np.argmax(q_values)

    def eps_scheduler(self):
        """
        eps_scheduler scheduler for calculating the epsilon depending on the episode_nr

        :return: epsilon value
        :rtype: float, [0 ... 1]
        """
        return 1/(self.eps_scheduler_rate * self.episode_nr + 1)

    def train(self, batch_size = 32):
        """
        train Train the Q model from the replay memory

        :param batch_size: Number of samples for the training, defaults to 32
        :type batch_size: int, optional
        """
        if len(self.replay_memory) < batch_size:
            batch_size = len(self.replay_memory)
        mini_batch = random.sample(self.replay_memory, batch_size - 1)
        mini_batch.append(self.replay_memory[0])


        current_states = np.array([mem[0] for mem in mini_batch])
        current_qs_list = self.q_model.predict(current_states)
        next_states = np.array([mem[3] for mem in mini_batch])
        next_qs_list = self.q_model.predict(next_states)

        States = [[0]*self.env.state_size] * batch_size
        Rewards = [[0]*self.env.action_size] * batch_size
        for index, (state, action, reward, next_state, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + self.discount_factor * np.max(next_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] =  max_future_q

            States[index] = state
            Rewards[index] = current_qs
        self.q_model.fit(np.array(States), np.array(Rewards), batch_size=batch_size, verbose=0, shuffle=True)



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

            if (steps % 10) or done :
                self.train()

            ts += self.env.time_step
            steps += 1
        self.episode_nr += 1
        return trajectory_data, total_training_rewards
