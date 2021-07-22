var socket;
var canvas_input;
var canvas_grid;
var ctx_grid;
var ctx_input;
var flag_pressed = false;
var prevX = 0;
var prevY = 0;
var currX = 0;
var currY = 0;
var first_pressed = true;

function init() {
    socket = io.connect();
    canvas_grid = document.getElementById("CanvasGrid");
    ctx_grid = canvas_grid.getContext("2d");
    canvas_input = document.getElementById("CanvasInput");
    ctx_input = canvas_input.getContext("2d");
    canvas_input.addEventListener("mousemove", function (e) {
        mouse_event('move', e)
    }, false);
    canvas_input.addEventListener("mousedown", function (e) {
        mouse_event('down', e)
    }, false);
    canvas_input.addEventListener("mouseup", function (e) {
        mouse_event('up', e)
    }, false);

    socket.emit("reset_event");
    drawGrid();
}

/**
 * Draw feasible area grid
 */
function drawGrid() {
    ctx_grid.fillStyle = "lightgrey";
    ctx_grid.fillRect(35, canvas_grid.height - 60, 500, 60);
    ctx_grid.beginPath();
    ctx_grid.arc(35, canvas_grid.height - 60, 500, 0, 1 * Math.PI, true);
    ctx_grid.fill();
    ctx_grid.fillStyle = "white";
    ctx_grid.fillRect(0, 0, 35, canvas_grid.height);
    ctx_grid.beginPath();
    ctx_grid.arc(35, canvas_grid.height - 60 - 250, 250, - Math.PI, 1 * Math.PI, true);
    ctx_grid.fill();

}

/**
 * Draw a line in Canvas
 */
function drawLine() {
    ctx_input.beginPath();
    ctx_input.moveTo(prevX, prevY);
    ctx_input.lineTo(currX, currY);
    ctx_input.strokeStyle = "black";
    ctx_input.lineWidth = 2;
    ctx_input.stroke();
    ctx_input.closePath();
}

/**
 * Callback if a mouse event occurs
 * @param {*} event The pressed event
 * @param {*} pos position of the mouse
 */
function mouse_event(event, pos) {
    var start = false;
    var x = pos.clientX - canvas_input.offsetLeft - canvas.offsetLeft;
    var y = pos.clientY - canvas_input.offsetTop - canvas.offsetTop;

    if (event == 'down') {
        flag_pressed = true;
        start = true;
        if (first_pressed) {
            currX = x;
            currY = y;
            prevX = x;
            prevY = y;
            first_pressed = false;
        }
    }
    if (event == 'up') {
        flag_pressed = false;
    }
    if (flag_pressed) {
        var milliseconds = new Date().getTime();
        socket.emit("draw_event", { pos: { x: x, y: y }, ts: milliseconds, start: start });
        prevX = currX;
        prevY = currY;
        currX = x;
        currY = y;
        drawLine()
    }
}

/**
 * Callback for Reset button
 */
function reset_canvas() {
    ctx_input.clearRect(0, 0, canvas_input.width, canvas_input.height);
    first_pressed = true;
    flag_pressed = false;
    socket.emit("reset_event");
    draw_robot(35 + 250, canvas_animation.height - 60, 35 + 250 * 2, canvas_animation.height - 60);
}

init();