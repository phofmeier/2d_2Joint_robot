from flask import Flask, render_template
from flask_socketio import SocketIO
from trajectory_planner.trajectoryData import ReferenceTrajectory, StateTrajectory
from trajectory_planner.trajectoryPlanner import TrajectoryPlanner
from trajectory_planner.environment import Environment
from trajectory_planner.deep_q_learning import DeepQLearning

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

# SocketIO event Listener
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

@socketio.on("start_learning")
def start_learning_callback():
    global output_trajectory
    input_trajectory.resample()
    reference = input_trajectory.getMetricDataArray()[:,1:]
    env = Environment(reference)
    learner = DeepQLearning(env)

    for i in range(100):
        output_trajectory, reward =  learner.runEpisode(canvas_width, canvas_height)
        print("Episode:", i, "Reward:", reward)

def main():
    socketio.run(app, port=port, host=host, debug=True)

if __name__ == "__main__":
    main()
