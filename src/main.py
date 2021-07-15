from time import sleep
from flask import Flask, render_template
from flask_socketio import SocketIO
from trajectoryData import ReferenceTrajectory, StateTrajectory
from trajectoryPlanner import TrajectoryPlanner

import copy


port = 5000
host = '127.0.0.1'

app = Flask(__name__,)
socketio = SocketIO(app)
canvas_width = 600
canvas_height = 400

input_trajectory = ReferenceTrajectory(canvas_width, canvas_height)
output_trajectory = StateTrajectory(canvas_width, canvas_height)


@app.route('/')
def index():
    input_trajectory.clear()
    return render_template('index.html', canvas_width=canvas_width, canvas_height=canvas_height)


@socketio.on('draw_event')
def draw_event_callback(argument):
    input_trajectory.addCanvasDataPoint(argument)

@socketio.on('reset_event')
def reset_callback():
    input_trajectory.clear()
    output_trajectory.clear()

@socketio.on('start_animation')
def animation_callback():
    socketio.emit("new_animation_value", output_trajectory.GetCanvasPositions())

@socketio.on('optimize_trajectory')
def optimize_trajectory_callback():
    global output_trajectory
    input_trajectory.resample()
    planner = TrajectoryPlanner(input_trajectory)
    solution = planner.plan()
    output_trajectory = StateTrajectory(canvas_width, canvas_height)
    output_trajectory.SetSolution(input_trajectory.getMetricDataArray()[:, 0], solution)
    socketio.emit("optimization_done")


if __name__ == "__main__":
    socketio.run(app, port=port, host=host, debug=True)
