from trajectory_planner.trajectoryData import ReferenceTrajectory
import casadi as cas
from trajectory_planner.model import TwoLinkModel


class TrajectoryPlanner:
    """
     This Class is for Planning an optimal trajectory
    """
    def __init__(self, reference_trajectory: ReferenceTrajectory):
        """
        __init__ Initialize the Planner with a reference trajectory

        :param reference_trajectory: The reference trajectory which should be followed
        :type reference_trajectory: ReferenceTrajectory
        """
        self.reference_trajectory = reference_trajectory

    def plan(self):
        """
        plan Plan a trajectory

        :return: Solution as supposed from the solver
        """
        solution = self.solve()
        return solution

    def solve(self):
        """
        solve build and solve the optimization problem

        :return: solution of the solver
        """
        model = TwoLinkModel()
        reference = self.reference_trajectory.getMetricDataArray()
        step_size = (reference[1, 0] - reference[0, 0]) * 1e-3

        # Define Variables
        w = []  # optimization variables
        w0 = []  # initial state
        lbw = []  # lower bound
        ubw = []  # upper bound
        J = 0  # cost function
        g = []  # constraint
        lbg = []  # lower bound constraint
        ubg = []  # upper bound constraint

        # Initial conditions
        # TODO(peter): make initial state changeable
        Xk = cas.MX.sym('X0', model.state_size)
        w += [Xk]
        lbw += model.state_lb
        ubw += model.state_ub
        w0 += [0] * model.state_size

        # Generate NLP for each step
        for k in range(self.reference_trajectory.length()):
            # Controls
            Uk = cas.MX.sym('U_' + str(k), model.control_size)
            w += [Uk]
            lbw += [-model.max_control] * model.control_size
            ubw += [model.max_control] * model.control_size
            w0 += [0] * model.control_size

            # New State
            Xk_end = model.discreteFun(Xk, Uk, step_size)
            Xk = cas.MX.sym('X_' + str(k+1), model.state_size)
            w += [Xk]
            lbw += model.state_lb
            ubw += model.state_ub
            w0 += [0] * model.state_size
            # State equality constraint
            g += [Xk_end-Xk]
            lbg += [0] * model.state_size
            ubg += [0] * model.state_size

            # Cost Function
            J = J + (cas.dot(reference[k, 1:] - model.calcPos2(Xk),
                     reference[k, 1:] - model.calcPos2(Xk))) + 0.000001 * cas.dot(Uk, Uk)

        # Create the Solver
        prob = {'f': J, 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
        solver = cas.nlpsol('solver', 'ipopt', prob)
        return solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
