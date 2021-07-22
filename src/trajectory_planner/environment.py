from trajectory_planner.model import TwoLinkModel
import numpy as np
import itertools


class Environment:

    def __init__(self, reference, model=TwoLinkModel()) -> None:
        self.time_step = 10 * 1e-3
        self.actionDiscreteSpace = 3
        self.model = model
        self.reference = reference
        self.currentModelState = self.model.initial_state
        self.currentStep = 0
        self.envState = [0, 0, 0, 0, 0, 0]
        action_space = np.linspace(-self.model.max_control,
                                   self.model.max_control, self.actionDiscreteSpace).tolist()
        self.actions = list(itertools.product(*[action_space] * self.model.control_size))
        self.action_size = len(self.actions)

    def reset(self):
        self.currentModelState = self.model.initial_state
        self.currentStep = 0
        self.envState = [0, 0, 0, 0, 0, 0]

    def step(self, action_index):
        current_env_state = self.envState
        current_model_state = self.currentModelState
        new_model_state = self.model.discreteFun(current_model_state, self.actions[action_index], self.time_step).full().flatten()
        reward = self.reward_function(self.reference[self.currentStep], new_model_state)
        self.currentModelState = new_model_state
        self.currentStep += 1
        done = self.currentStep > len(self.reference)
        new_env_state = self.getCurrentEnvState()
        self.envState = new_env_state

        return current_env_state, action_index, reward, new_env_state, done

    def reward_function(self, reference, state):
        pos_state = self.model.calcPos2_np(state[0], state[2])
        diff = reference - pos_state.flatten()
        return np.dot(diff, diff)

    def getCurrentEnvState(self):
        pos_state = self.model.calcPos2_np(self.currentModelState[0], self.currentModelState[2]).flatten()
        e_x = self.reference[self.currentStep][0] - pos_state[0]
        e_y = self.reference[self.currentStep][1] - pos_state[1]

        return np.concatenate((self.currentModelState, [e_x, e_y]))
