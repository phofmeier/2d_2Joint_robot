import pytest
from trajectory_planner.environment import Environment


def test_envTest():
    reference = [[1, 2], [3, 4], [5, 6]]
    env = Environment(reference)
    env.reset()
    current_env_state, action, reward, new_env_state, done = env.step(0)
    assert current_env_state.tolist() == [0, 0, 0, 0, 0, 2.0]
    assert action == 0
    assert reward == -4.003000562124957
    assert new_env_state.tolist() == pytest.approx([-5.00000000e-04, -1.00000000e-01,
                                                    -5.00000000e-04, -1.00000000e-01,
                                                    2.00000031e+00,  4.00075000e+00])
    assert done == False

    current_env_state, action, reward, new_env_state, done = env.step(0)
    assert current_env_state.tolist() == pytest.approx([-5.00000000e-04, -1.00000000e-01,
                                                        -5.00000000e-04, -1.00000000e-01,
                                                        2.00000031e+00,  4.00075000e+00])
    assert action == 0
    assert reward == -20.024028951966365
    assert new_env_state.tolist() == pytest.approx([-0.0020000000000000005, -0.2,
                                                    -0.0020000000000000005, -0.2,
                                                    4.000004999994333, 6.002999994000004])
    assert done == False

    current_env_state, action, reward, new_env_state, done = env.step(0)
    assert current_env_state.tolist() == pytest.approx([-0.0020000000000000005, -0.2,
                                                        -0.0020000000000000005, -0.2,
                                                        4.000004999994333, 6.002999994000004])
    assert action == 0
    assert reward == -52.081247240934296
    assert new_env_state.tolist() == pytest.approx([-0.0045000000000000005, -0.30000000000000004,
                                                    -0.0045000000000000005, -0.30000000000000004,
                                                    0.0, 0.0])
    assert done == True
