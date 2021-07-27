from typing import Dict
from numpy import uint64
from pandas import DataFrame
from scipy.interpolate import interp1d
import numpy as np
from trajectory_planner.model import TwoLinkModel


class ReferenceTrajectory:
    """
     This class hold the data for one Position reference trajectory  and provides some basic functions

    """

    def __init__(self, canvas_width: int, canvas_height: int) -> None:
        """
        __init__ Initialize the Data container and constants

        :param canvas_width: width of the canvas object
        :type canvas_width: int
        :param canvas_height: height of the canvas object
        :type canvas_height: int
        """
        self.data = DataFrame(columns=["ts", "pos_x", "pos_y"])
        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.canvas_zero_x = 35
        self.canvas_zero_y = canvas_height - 60
        self.canvas_scale = 500
        self.periode_ms = 10

    def length(self):
        """
        length Length of the trajectory

        :return: length
        """
        return len(self.data)

    def addCanvasDataPoint(self, data: Dict) -> None:
        """
        addCanvasDataPoint Add a new Data point in canvas coordinate system

        :param data: Dict containing {timestamp, pos:{x,y}}
        :type data: Dict
        """
        metric = self.canvasToMetric([float(data["pos"]["x"]), float(data["pos"]["y"])])
        self.data = self.data.append(
            {"ts": uint64(data["ts"]), "pos_x": metric[0], "pos_y": metric[1]}, ignore_index=True)

    def resample(self):
        """
        resample resample the trajectory using linear interpolation and set start timestamp to 0
        """
        if self.data.empty:
            return
        first_ts = self.data["ts"].iloc[0]
        last_ts = self.data["ts"].iloc[-1]
        new_ts = np.arange(first_ts, last_ts, self.periode_ms)
        pos_x_interpolated = interp1d(self.data["ts"], self.data["pos_x"])(new_ts)
        pos_y_interpolated = interp1d(self.data["ts"], self.data["pos_y"])(new_ts)
        new_ts = new_ts - first_ts
        self.data = DataFrame({"ts": new_ts, "pos_x": pos_x_interpolated,
                              "pos_y": pos_y_interpolated})

    def getCanvasData(self) -> Dict:
        """
        getCanvasData get the Trajectory in Canvas coordinate system

        :return: Dict containing timestamps and x and y position
        :rtype: Dict
        """
        metric = self.data.to_numpy(copy=True)
        canvas = metric
        canvas[:, 1:] = self.metricToCanvas(metric[:, 1:])
        return DataFrame(canvas, columns=["ts", "pos_x", "pos_y"]).to_dict(orient="records")

    def getMetricDataArray(self):
        """
        getMetricDataArray get a numpy array with the trajectory as metric Data

        :return: array containing tiemstamp x and y position
        :rtype: np.array
        """
        return self.data.to_numpy(copy=True)

    def clear(self):
        """
        clear Clear the trajectory
        """
        self.data = DataFrame(columns=["ts", "pos_x", "pos_y"])

    def canvasToMetric(self, pos):
        """
        canvasToMetric Convert from canvas to metric coordinate system

        :param pos: x, y Position in canvas coordinate system
        :type pos: List
        :return: x,y position in metric coordinate system
        :rtype: List
        """
        x_metric = (pos[0] - self.canvas_zero_x) / self.canvas_scale
        y_metric = (self.canvas_zero_y - pos[1]) / self.canvas_scale
        return [x_metric, y_metric]

    def metricToCanvas(self, pos):
        """
        metricToCanvas Convert from metric to canvas coordinate system

        :param pos: x,y in metric coordinate system
        :type pos: np.array
        :return: x,y in canvas coordinate system
        :rtype: np.array
        """
        x_canvas = pos[:, 0] * self.canvas_scale + self.canvas_zero_x
        y_canvas = self.canvas_zero_y - (pos[:, 1] * self.canvas_scale)
        return np.column_stack((x_canvas, y_canvas))


class StateTrajectory:
    """
     This Class hold a planned trajectory of states an Controls
    """

    def __init__(self, canvas_width: int, canvas_height: int) -> None:
        self.model = TwoLinkModel()
        self.data = DataFrame(columns=['ts', 'x_0', 'x_1', 'x_2', 'x_3', 'u_0', 'u_1'])

        self.canvas_width = canvas_width
        self.canvas_height = canvas_height
        self.canvas_zero_x = 35
        self.canvas_zero_y = canvas_height - 60
        self.canvas_scale = 500

    def SetSolution(self, ts, solution):
        """
        SetSolution Generate a Trajectory from a solution of a solver

        :param ts: Timestamps
        :param solution: solution of the solver
        """
        self.data = DataFrame()
        w_opt = solution['x'].full().flatten()
        w_opt_size = self.model.state_size + self.model.control_size
        for i in range(self.model.state_size):
            self.data.insert(i, "x_"+str(i), w_opt[i::w_opt_size])
        for i in range(self.model.control_size):
            self.data.insert(i+self.model.state_size, "u_"+str(i),
                             np.append(w_opt[i+self.model.state_size::w_opt_size], np.nan))
        self.data.insert(0, "ts", np.append(ts, ts[-1] + (ts[1]-ts[0])))

    def AppendState(self, x, u, ts):
        """
        AppendState Append one state to the trajectory

        :param x: state
        :param u: controls
        :param ts: timestamp
        """
        df = DataFrame([[ts, x[0], x[1], x[2], x[3], u[0], u[1]]],columns=['ts', 'x_0', 'x_1', 'x_2', 'x_3', 'u_0', 'u_1'] )
        self.data = self.data.append(df, ignore_index=True)

    def GetCanvasPositions(self):
        """
        GetCanvasPositions Get the Positions and Timestamps in the Canvas coordination system

        :return: [ts, x_1, y_1, x_2, y_2] in canvas coordination system
        :rtype: [type]
        """
        pos_1 = self.model.calcPos1_np(self.data["x_0"].to_numpy())
        pos_2 = self.model.calcPos2_np(self.data["x_0"].to_numpy(), self.data["x_2"].to_numpy())
        pos_1_canvas = self.metricToCanvas(pos_1.transpose())
        pos_2_canvas = self.metricToCanvas(pos_2.transpose())

        canvas_data = DataFrame()
        canvas_data.insert(0, "ts", self.data["ts"])
        canvas_data.insert(1, "x_1", pos_1_canvas[:, 0])
        canvas_data.insert(2, "y_1", pos_1_canvas[:, 1])
        canvas_data.insert(3, "x_2", pos_2_canvas[:, 0])
        canvas_data.insert(4, "y_2", pos_2_canvas[:, 1])
        return canvas_data.to_dict(orient="records")

    def metricToCanvas(self, pos):
        """
        metricToCanvas Convert from metric to canvas coordinate system

        :param pos: x,y in metric coordinate system
        :type pos: np.array
        :return: x,y in canvas coordinate system
        :rtype: np.array
        """
        x_canvas = pos[:, 0] * self.canvas_scale + self.canvas_zero_x
        y_canvas = self.canvas_zero_y - (pos[:, 1] * self.canvas_scale)
        return np.column_stack((x_canvas, y_canvas))

    def clear(self):
        """
        clear Clear the trajectory
        """
        self.data = DataFrame()
