from flask import Flask, render_template
from flask_socketio import SocketIO
from trajectory_planner.trajectoryData import ReferenceTrajectory, StateTrajectory
from trajectory_planner.trajectoryPlanner import TrajectoryPlanner
from trajectory_planner.environment import Environment
from trajectory_planner.deep_q_learning import DeepQLearning
from multiprocessing import Process, Queue
import time

port = 5000
host = '127.0.0.1'

app = Flask(__name__,)
socketio = SocketIO(app)
canvas_width = 600
canvas_height = 400

input_trajectory = ReferenceTrajectory(canvas_width, canvas_height)
output_trajectory = StateTrajectory(canvas_width, canvas_height)
q = Queue(maxsize=20)
learned_trajectories = []



@app.route('/')
def index():
    input_trajectory.clear()
    return render_template('index.html', canvas_width=canvas_width, canvas_height=canvas_height)

# SocketIO event Listener

@socketio.on("timer")
def timer_callback():
    global output_trajectory, q, learned_trajectories
    while not q.empty():
        value = q.get(block=False)
        output_trajectory = value[0]
        learned_trajectories.append(value)
        socketio.emit("new_learned_reward", [value[2], value[1]])



@socketio.on('draw_event')
def draw_event_callback(argument):
    input_trajectory.addCanvasDataPoint(argument)


@socketio.on('reset_event')
def reset_callback():
    input_trajectory.clear()
    output_trajectory.clear()

@socketio.on("start_animation_index")
def animation_k_callback(index):
    socketio.emit("new_animation_value", learned_trajectories[int(index["index"])][0].GetCanvasPositions())

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
    global q

    p = Process(target=learn, args=(q,), daemon=True)
    p.start()


def learn(q):
    path = "trained_model/" + time.strftime("%Y%m%d-%H%M%S") + "/Episode_"
    input_trajectory.resample()
    reference = input_trajectory.getMetricDataArray()[:, 1:]
    env = Environment(reference)
    # TODO mit weniger actions, laengerer reference, nur reference und nicht den fehler
    learner = DeepQLearning(env, learning_rate=0.001, discount_factor=0.999,
                            N_hidden_layer=4, layer_size=64, eps_scheduler_rate=200, batch_size=512)
    learner.generate_random_data(20)
    learner.fit_normalizer()

    for i in range(2000):
        out_trajectory, reward = learner.runEpisode(canvas_width, canvas_height)
        print("Episode:", i, "Reward:", reward)
        if q.full():
            q.get()
        q.put([out_trajectory, reward, i])
        if (i%100) == 0:
            learner.save_model(path + str(i))



def main():
    socketio.run(app, port=port, host=host, debug=False)


if __name__ == "__main__":
    main()
