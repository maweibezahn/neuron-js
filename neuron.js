/* 
 * This is the simplest possible version of a neural net. In fact, it is only one neuron.
 * It expects 3 input values and puts out one output value between 0 and 1
 * 
 * This is heavilly inspired by these tutorials:
 * https://medium.com/technology-invention-and-more/how-to-build-a-simple-neural-network-in-9-lines-of-python-code-cc8f23647ca1#.cttmfd3tv
 * http://iamtrask.github.io/2015/07/12/basic-python-network/
 */


// train the network and test it with some random values
var init = function() {
    updateOutput("Random starting synaptic weight: " + synapticWeights);
    trainingPhase();
    updateOutput("New synaptic weight: " + synapticWeights);
    test([Math.round(Math.random()),
        Math.round(Math.random()),
        Math.round(Math.random())
    ]);
};


// test the network
var test = function(input) {
    var output = calculateOutput(input, synapticWeights);
    updateOutput("For input: " + input + " the output is: " + output);
};


// log and show results in the browser window
var updateOutput = function(text) {
    document.body.innerHTML = text;
    console.log(text);
};


// train the network for 100000 times.
// The rule: if the first number is 1, the output is 1, else it is 0
var trainingPhase = function() {
    var inputs = [
        [0, 0, 1],
        [1, 1, 1],
        [1, 0, 1],
        [0, 1, 1]
    ];
    var outputs = [0, 1, 1, 0];
    train(inputs, outputs, 100000);
};


// the weights, a value between -1 and 1. The amount of weights must match the number of input values
// The weights tell the neuron how important a certain input is. We will change these weights through the training phase
var synapticWeights = [Math.random() * 2 - 1, Math.random() * 2 - 1, Math.random() * 2 - 1];


// train the neuron for a given number of rounds
var train = function(inputs, outputs, iterations) {
    for (var i = 0; i < iterations; i++) {
        for (var j = 0; j < inputs.length; j++) {
            var singleInputs = inputs[j];
            // here we will always get a value between 0 and 1 (normalized)
            var output = calculateOutput(singleInputs, synapticWeights);
            // the error is simply the difference between our expected output and the current output
            var error = outputs[j] - output;
            // tells us by how much to change the weights for the next round
            var gradient = derivative(output);
            for (var k = 0; k < synapticWeights.length; k++) {
                // make adjustments for every weight. If the input is 0 the weight will not change
                var adjust = error * gradient * singleInputs[k];
                synapticWeights[k] += adjust;
            }
        }
    }
};


// the output is the normalized (using the sigmoid function) product of the weighted inputs
var calculateOutput = function(inputs, weights) {
    var ret = 0;
    for (var i = 0; i < inputs.length; i++) {
        ret += inputs[i] * weights[i];
    }
    ret = sigmoid(ret);
    return ret;
};


// the sigmoid function normalizes a value, the output is always between 0 and 1
// the reason we use this function is that it will amplify the difference between high or low values and
// values in between
// https://en.wikipedia.org/wiki/Sigmoid_function
var sigmoid = function(x) {
    return 1 / (1 + Math.exp(-x));
};


// the derivative of a function (in our case we use the output of the sigmoid function)
// this way we can get the slope of the function wich we use to alternate our weights for the next round
var derivative = function(x) {
    return x * (1 - x);
};