from trajectoryData import ReferenceTrajectory, StateTrajectory
import copy
import casadi as cas
from model import TwoLinkModel


class TrajectoryPlanner:
    def __init__(self, reference_trajectory: ReferenceTrajectory) -> None:
        self.reference_trajectory = reference_trajectory

    def plan(self):
        solution = self.solve()
        return solution

    def solve(self):
        model = TwoLinkModel()
        step_size = 10*1e-3
        reference = self.reference_trajectory.getMetricDataArray()

        # Start with an empty NLP
        w = []  # optimization variables
        w0 = []  # initial state
        lbw = []  # lower bound
        ubw = []  # upper bound
        J = 0  # cost function
        g = []  # constraint
        lbg = []  # lower bound constraint
        ubg = []  # upper bound constraint

        # "Lift" initial conditions
        # TODO(peter): make initial state changeable
        Xk = cas.MX.sym('X0', model.state_size)
        w += [Xk]
        lbw += model.state_lb
        ubw += model.state_ub
        w0 += [0] * model.state_size

        # Formulate the NLP
        for k in range(self.reference_trajectory.length()):
            # New NLP variable for the control
            Uk = cas.MX.sym('U_' + str(k), model.control_size)
            w += [Uk]
            lbw += [-model.max_control] * model.control_size
            ubw += [model.max_control] * model.control_size
            w0 += [0] * model.control_size

            # Integrate till the end of the interval
            Xk_end = model.discreteFun(Xk, Uk, step_size)

            # New NLP variable for state at end of interval
            Xk = cas.MX.sym('X_' + str(k+1), model.state_size)
            w += [Xk]
            lbw += model.state_lb
            ubw += model.state_ub
            w0 += [0] * model.state_size

            J = J + (cas.dot(reference[k, 1:] - model.calcPos2(Xk),
                     reference[k, 1:] - model.calcPos2(Xk))) + 0.000001 * cas.dot(Uk, Uk)

            # Add equality constraint
            g += [Xk_end-Xk]
            lbg += [0] * model.state_size
            ubg += [0] * model.state_size

        # Create an NLP solver
        prob = {'f': J, 'x': cas.vertcat(*w), 'g': cas.vertcat(*g)}
        solver = cas.nlpsol('solver', 'ipopt', prob)
        return solver(x0=w0, lbx=lbw, ubx=ubw, lbg=lbg, ubg=ubg)
