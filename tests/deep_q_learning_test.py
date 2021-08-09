from trajectory_planner.environment import Environment
from trajectory_planner.deep_q_learning import DeepQLearning
import numpy as np
import cProfile

def test_deep_q_learning():
    reference = [[1, 2], [3, 4], [5, 6]]
    env = Environment(reference)
    learner = DeepQLearning(env)
    action = learner.eps_greedy(1, [[0, 0, 0, 0, 0, 0]])
    print("eps=1:", action)

    action = learner.eps_greedy(0, [[0, 0, 0, 0, 0, 0]])
    assert action == 0

    learner.replay_memory.append([np.array([0, 0, 0, 0, 0, 0]), 0, 10,
                                 np.array([1, 0, 0, 0, 0, 0]), False])
    learner.replay_memory.append([np.array([0, 1, 0, 0, 0, 0]), 1, 11,
                                 np.array([0, 0, 1, 0, 0, 0]), False])
    learner.replay_memory.append([np.array([0, 0, 0, 1, 0, 0]), 2, 12,
                                 np.array([0, 0, 0, 0, 1, 0]), True])

    learner.train()
    assert False


def test_run_episode():
    reference = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [
        1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
    env = Environment(reference)
    learner = DeepQLearning(env, eps_scheduler_rate=20)

    for i in range(2):
        learner.runEpisode(600,400)

    assert False


# from trajectory_planner.environment import Environment
# from trajectory_planner.deep_q_learning import DeepQLearning
# reference = [[1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [
#     1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0], [1, 0]]
# env = Environment(reference)
# learner = DeepQLearning(env)

# for i in range(10):
#     learner.runEpisode(600, 400)

