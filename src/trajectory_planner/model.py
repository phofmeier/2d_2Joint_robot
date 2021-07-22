import numpy as np
import casadi as cas


def rk4(ode, x, u, h):
    """
    rk4 Runge-Kutta-4 Integrator

    :param ode: differential equation
    :param x: State
    :param u: Controls(zero order hold)
    :param h: step size
    :return: next state after integration
    """
    k1 = ode(x, u)
    k2 = ode(x + (h / 2.0) * k1, u)
    k3 = ode(x + (h / 2.0) * k2, u)
    k4 = ode(x + h * k3, u)
    x_next = x + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

    return x_next


class TwoLinkModel:
    """
    Simple Model of 2 Dimensional Robotic arm with 2 actuated joints

    state x = [alpha_1, d/dt alpha_1, alpha_2, d/dt alpha_2]
    control u = [u_1, u_2]
    """

    def __init__(self) -> None:
        self.l1 = 0.5  # Length of first arm [m]
        self.l2 = 0.5  # Length of second arm [m]
        self.state_size = 4
        self.control_size = 2
        self.initial_state = [0, 0, 0, 0]

        # Constraints
        self.max_control = 10
        self.state_lb = [0, -cas.inf, -3.1, -cas.inf]
        self.state_ub = [np.pi/2, cas.inf, 3.1, cas.inf]


    def calcPos1(self, x):
        """
        calcPos1 Calculate Position of first Joint

        :param x: state
        :return: Position [x, y]
        """
        x_1 = self.l1 * cas.cos(x[0])
        y_1 = self.l1 * cas.sin(x[0])
        return cas.vertcat(x_1, y_1)

    def calcPos1_np(self, alpha_1):
        """
        calcPos1_np Calculate Position of first joint from first angle

        :param alpha_1: First angle in rad
        :return: position [x, y]
        """
        x_1 = self.l1 * np.cos(alpha_1)
        y_1 = self.l1 * np.sin(alpha_1)
        return np.array([x_1, y_1])

    def calcPos2(self, x):
        """
        calcPos2 Calculate Position of second Joint

        :param x: state
        :return: Position [x, y]
        """
        p_1 = self.calcPos1(x)
        x_2 = p_1[0] + self.l2 * cas.cos(x[0] + x[2])
        y_2 = p_1[1] + self.l2 * cas.sin(x[0] + x[2])
        return cas.vertcat(x_2, y_2)

    def calcPos2_np(self, alpha_1, alpha_2):
        """
        calcPos2_np Calculate Position of second joint from first and second angle

        :param alpha_1: First angle in rad
        :param alpha_2: second angle in rad
        :return: Position [x, y]
        """
        p_1 = self.calcPos1_np(alpha_1)
        x_2 = p_1[0] + self.l2 * np.cos(alpha_1 + alpha_2)
        y_2 = p_1[1] + self.l2 * np.sin(alpha_1 + alpha_2)
        return np.array([x_2, y_2])

    def ode(self, x, u):
        """
        ode Ordinary Differential Equation

        :param x: State
        :param u: Controls
        :return: ode
        """
        d1alpha_1 = x[1]
        d2alpha_1 = u[0]
        d1alpha_2 = x[3]
        d2alpha_2 = u[1]
        return cas.vertcat(d1alpha_1, d2alpha_1, d1alpha_2, d2alpha_2)

    def discreteFun(self, x, u, h):
        """
        discreteFun Discrete Function of the ode (x_t+h = F(x_t, u_t))

        :param x: state
        :param u: controls
        :param h: step size
        :return: next state
        """
        return rk4(self.ode, x, u, h)
