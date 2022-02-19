var socket;
var timeout = 1000;
var rewards = [];
var steps = [];

function init() {
    socket = io.connect();

    socket.on("new_learned_reward", (reward_input) => {
        window.rewards.push(reward_input[1]);
        window.steps.push(reward_input[0]);
        update_figure();
        slider = document.getElementById("step_slider")
        slider.max = reward_input[0];
    });
    update_figure();
    window.setInterval(function(){ socket.emit("timer"); }, 1000)

}

function SliderChanged(val) {
    document.getElementById('slider_value').value = val;
    update_figure();
}

function animate_k() {
    socket.emit("start_animation_index", { index: document.getElementById('slider_value').value});
    window.animation_index = 0;
}


function update_figure() {
    slider = document.getElementById("step_slider")
    var layout = {
        title: 'Cost per Episode',
        xaxis: {
          title: 'Episode Nr.'
        },
        yaxis: {
            title: 'Total Cost',
            type: 'log',
            autorange: true
        },
        shapes: [{
            type: 'line',
            x0: slider.value,
            y0: 0,
            x1: slider.value,
            yref: 'paper',
            y1: 1,
            line: {
                color: 'grey',
                width: 1.5,
                dash: 'dot'
            }
        }],
        autosize: false,
        hovermode:'closest',
        width: 1200,
        height: 600,
      };
    reward_plot = document.getElementById('reward_plot');
    Plotly.newPlot(reward_plot, [{
        x: steps,
        y: rewards
    }], layout);

    reward_plot.on('plotly_click', function(data){
        slider = document.getElementById("step_slider");
        slider.value = data.points[0].x;
        SliderChanged(slider.value)
    });

}

/**
 * Callback for Reset button
 */
 function start_learning() {
    socket.emit("start_learning");
 }

 init();