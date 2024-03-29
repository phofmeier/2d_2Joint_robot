from trajectory_planner.model import TwoLinkModel
import numpy as np
import itertools


class Environment:
    """
     This class behaves as a wrapper for the model which is used in the Q Learning algorithm
    """

    def __init__(self, reference, model=TwoLinkModel()) -> None:
        """
        __init__ Initialize the class

        :param reference: reference trajectory
        :param model: The state space Modelclass for the environment, defaults to TwoLinkModel()
        :type model: ModelClass, optional
        """
        self.done = False
        self.time_step = 10 * 1e-3
        self.actionDiscreteSpace = 5
        self.model = model
        self.reference = reference
        self.currentModelState = self.model.initial_state
        self.currentStep = 0
        self.envState = self.getCurrentEnvState()
        self.state_size = len(self.envState)
        action_space = np.linspace(-self.model.max_control,
                                   self.model.max_control, self.actionDiscreteSpace).tolist()
        self.actions = list(itertools.product(*[action_space] * self.model.control_size))
        self.action_size = len(self.actions)

    def reset(self):
        """
        reset reset the Environment to initial state

        :return: initial environment state
        """
        self.currentModelState = self.model.initial_state
        self.currentStep = 0
        self.done = False
        self.envState = self.getCurrentEnvState()
        return self.envState

    def calculateNewState(self, action_index):
        """
        calculateNewState Calculate new internal state

        :param action_index: index of the action to apply
        :return: new model state at next time step
        """
        new_state = self.model.discreteFun(self.currentModelState, self.actions[action_index], self.time_step).full().flatten()
        # Apply lower and upper bound to the states
        if new_state[0] < self.model.state_lb[0]:
            new_state[0] = self.model.state_lb[0]
            new_state[1] = 0
        elif new_state[0] > self.model.state_ub[0]:
            new_state[0] = self.model.state_ub[0]
            new_state[1] = 0

        if new_state[2] < self.model.state_lb[2]:
            new_state[2] = self.model.state_lb[2]
            new_state[3] = 0
        elif new_state[2] > self.model.state_ub[2]:
            new_state[2] = self.model.state_ub[2]
            new_state[3] = 0
        return new_state


    def step(self, action_index):
        """
        step Make one step forward in time

        :param action_index: index of the action to apply
        :return: current env state, action index, reward, new env state, done
        """
        current_env_state = self.envState
        new_model_state = self.calculateNewState(action_index)
        reward = self.reward_function(self.reference[self.currentStep], new_model_state, self.actions[action_index])
        self.currentModelState = new_model_state
        self.currentStep += 1
        self.done = self.currentStep >= len(self.reference)
        new_env_state = self.getCurrentEnvState()
        self.envState = new_env_state

        return current_env_state, action_index, reward, new_env_state, self.done

    def reward_function(self, reference, state, action):
        """
        reward_function Reward Function

        :param reference: reference point
        :param state: state
        :return: reward for reference and state
        """
        pos_state = self.model.calcPos2_np(state[0], state[2])
        diff = reference - pos_state.flatten()
        error = np.dot(diff, diff)

        # if self.currentStep >= len(self.reference):
        #     error = error * 1e3
        c = 1/ len(self.reference)
        # mu = 0.01
        # w = np.arctanh(np.sqrt(0.95)) / mu
        # cost = np.tanh(np.abs(error) * w) ** 2 * c        

        return c * (error + 1e-6 * np.dot(action, action))
        # return cost

    def getCurrentEnvState(self):
        """
        getCurrentEnvState get the Current environment state from model state and reference

        :return: environment state
        """
        
        pos_state = self.model.calcPos2_np(self.currentModelState[0], self.currentModelState[2]).flatten()
        # if self.done:
        #     x_ref = self.reference[self.currentStep - 1][0]
        #     y_ref = self.reference[self.currentStep - 1][1]

        # else:
        #     x_ref = self.reference[self.currentStep][0]
        #     y_ref = self.reference[self.currentStep][1]

        # x_cur = pos_state[0]
        # y_cur = pos_state[1]
        # return np.concatenate((self.currentModelState, [x_cur, y_cur ,x_ref, y_ref]))

        nr_ref_states = 40
        if self.done:
            x_ref = self.reference[self.currentStep - 1 :][:,0]
            y_ref = self.reference[self.currentStep - 1 :][:,1]

        else:
            x_ref = self.reference[self.currentStep :][:,0]
            y_ref = self.reference[self.currentStep :][:,1]

        
        if (len(x_ref) >= nr_ref_states):
            x_ref = x_ref[:nr_ref_states]
            y_ref = y_ref[:nr_ref_states]
        else :
            x_ref = np.concatenate((x_ref, np.ones(nr_ref_states-len(x_ref)) * x_ref[-1]))
            y_ref = np.concatenate((y_ref, np.ones(nr_ref_states-len(y_ref)) * y_ref[-1]))


        x_err = pos_state[0] - x_ref
        y_err = pos_state[1] - y_ref
        return np.concatenate((self.currentModelState,pos_state, x_ref, y_ref)).flatten()
