var socket;
var canvas_animation;
var ctx_animation;
var positions;
var animation_index = 0;


function init() {
    socket = io.connect();
    canvas_animation = document.getElementById("CanvasAnimation");
    ctx_animation = canvas_animation.getContext("2d");
    draw_robot(100, 100, 200, 200);

    socket.on('new_animation_value', (positions_input) => {
        window.positions = positions_input;
        animation();
    });

    socket.on('optimization_done', () => {
        button = document.getElementById("optimize_button");
        button.disabled = false;
    });
}

function draw_robot(x_1, y_1, x_2, y_2) {
    ctx_animation.clearRect(0, 0, canvas_animation.width, canvas_animation.height);

    // Draw Base
    ctx_animation.fillStyle = "black";
    ctx_animation.fillRect(10, canvas_animation.height - 60, 50, 50);
    ctx_animation.beginPath();
    ctx_animation.arc(35, canvas_animation.height - 60, 25, 0, 2 * Math.PI);
    ctx_animation.fill();

    // Draw first Part
    ctx_animation.fillStyle = "red";
    ctx_animation.beginPath();
    ctx_animation.arc(x_1, y_1, 15, 0, 2 * Math.PI);
    ctx_animation.fill();

    ctx_animation.beginPath();
    ctx_animation.moveTo(10 + 25, canvas_animation.height - 60);
    ctx_animation.lineTo(x_1, y_1);
    ctx_animation.strokeStyle = "red";
    ctx_animation.lineWidth = 10;
    ctx_animation.stroke();
    ctx_animation.closePath();
    ctx_animation.beginPath();
    ctx_animation.arc(10 + 25, canvas_animation.height - 60, 5, 0, 2 * Math.PI);
    ctx_animation.fill();

    // Draw second part
    ctx_animation.fillStyle = "green";
    ctx_animation.beginPath();
    ctx_animation.arc(x_2, y_2, 10, 0, 2 * Math.PI);
    ctx_animation.fill();

    ctx_animation.beginPath();
    ctx_animation.moveTo(x_1, y_1);
    ctx_animation.lineTo(x_2, y_2);
    ctx_animation.strokeStyle = "green";
    ctx_animation.lineWidth = 10;
    ctx_animation.stroke();
    ctx_animation.closePath();
    ctx_animation.beginPath();
    ctx_animation.arc(x_1, y_1, 5, 0, 2 * Math.PI);
    ctx_animation.fill();
}

function animation() {
    if (positions.length == 0) {
        return;
    }
    x_1_i = positions[animation_index]["x_1"];
    y_1_i = positions[animation_index]["y_1"];
    x_2_i = positions[animation_index]["x_2"];
    y_2_i = positions[animation_index]["y_2"];
    draw_robot(x_1_i, y_1_i, x_2_i, y_2_i);
    window.animation_index = animation_index + 1;

    if (animation_index < positions.length - 1) {
        timeout = positions[animation_index]["ts"] - positions[animation_index - 1]["ts"];
        window.setTimeout(animation, timeout);
    }
}

function animate_robot() {
    socket.emit("start_animation");
    window.animation_index = 0;
}

function optimize_trajectory() {
    button = document.getElementById("optimize_button");
    button.disabled = true;
    socket.emit("optimize_trajectory");
}


init();
