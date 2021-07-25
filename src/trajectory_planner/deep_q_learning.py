from trajectory_planner.trajectoryData import StateTrajectory
from trajectory_planner.environment import Environment
import tensorflow as tf
import numpy as np
from collections import deque
import random

class DeepQLearning:

    def __init__(self, environment) -> None:
        self.env = environment
        self.learning_rate = 0.01
        self.discount_factor = 0.618
        self.N_hidden_layer = 2
        self.layer_size = 24
        self.q_model = self.generate_q_model(self.N_hidden_layer, self.layer_size)
        self.replay_memory = deque(maxlen=100_000)

        self.episode_nr = 0

    def generate_q_model(self, N_hidden_layer=1, layer_size=24):
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
        random_number = np.random.rand()
        if random_number <= eps:
            # Exploration
            return np.random.randint(0, self.env.action_size)

        # Exploitation
        q_values = self.q_model.predict(state.reshape((1, self.env.state_size)))
        return np.argmax(q_values)

    def eps_scheduler(self):
        return 1/(0.1 * self.episode_nr + 1)

    def train(self, batch_size = 32):
        if len(self.replay_memory) < batch_size:
            batch_size = len(self.replay_memory)
        mini_batch = random.sample(self.replay_memory, batch_size - 1)
        mini_batch.append(self.replay_memory[0])


        current_states = np.array([mem[0] for mem in mini_batch])
        current_qs_list = self.q_model.predict(current_states)
        next_states = np.array([mem[3] for mem in mini_batch])
        next_qs_list = self.q_model.predict(next_states)

        States = []
        Rewards = []
        for index, (state, action, reward, next_state, done) in enumerate(mini_batch):
            if not done:
                max_future_q = reward + self.discount_factor * np.max(next_qs_list[index])
            else:
                max_future_q = reward

            current_qs = current_qs_list[index]
            current_qs[action] = max_future_q

            States.append(state)
            Rewards.append(current_qs)
        self.q_model.fit(np.array(States), np.array(Rewards), batch_size=batch_size, verbose=0, shuffle=True)



    def runEpisode(self, canvas_width, canvas_height):
        trajectory_data = StateTrajectory(canvas_width, canvas_height)
        total_training_rewards = 0
        current_state = self.env.reset()
        done = False
        ts = 0

        while not done:
            action = self.eps_greedy(self.eps_scheduler(), current_state)
            next_step = self.env.step(action)
            trajectory_data.AppendState(current_state[:4], self.env.actions[action] ,ts)
            self.replay_memory.append(next_step)
            current_state = next_step[3]
            total_training_rewards += next_step[2]

            self.train()

            done = next_step[4]
            ts += self.env.time_step
        self.episode_nr += 1
        return trajectory_data, total_training_rewards