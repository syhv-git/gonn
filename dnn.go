package gonn

import (
	"gonum.org/v1/gonum/floats"
	"log"
	"math/rand"
	"time"
)

type DNN struct {
	Rate float64

	activation, variance ActivationFunc

	loss, lossPrime LossFunc

	hidden []*dnnLayer

	output *dnnLayer
}

// NewDNN creates a new Dropout Multilayer Perceptron
//
// sizes contains the sizes of each layer in the network
//
// learningRate is the scaling hyperparameter for gradient descent
//
// activation defines the normalization function applied to the output
//
// variance defines the derivative of the activation function used
func NewDNN(sizes []int, learningRate float64, activation, variance ActivationFunc, loss, lossPrime LossFunc) *DNN {
	if len(sizes) < 2 {
		panic("invalid slices parameter supplied to NewDNN")
	}

	layers, output := make([]*dnnLayer, len(sizes)-2), newDropoutLayer(sizes[len(sizes)-2], sizes[len(sizes)-1])
	for i := 1; i <= len(layers); i++ {
		layers[i-1] = newDropoutLayer(sizes[i-1], sizes[i])
	}

	ret := &DNN{Rate: learningRate, activation: activation, variance: variance, loss: loss, lossPrime: lossPrime, hidden: layers, output: output}
	return ret
}

// SetDropout sets the dropout rate for the network.
//
// rate defines the dropout rate for the network. It can contain one value for the whole network or specific values
// for each layer in the network.
//
// ** The dropout rate must be set before training, and then set to 0 before predicting new data **
func (nn *DNN) SetDropout(rate []float64) {
	if len(rate) != 1 && len(rate) != len(nn.hidden)+1 {
		log.Fatal("supplied dropout rates are not compatible with this network")
	}

	for i := range nn.hidden {
		if len(rate) > 1 {
			nn.hidden[i].setDropout(rate[i])
		} else {
			nn.hidden[i].setDropout(rate[0])
		}
	}

	if len(rate) > 1 {
		nn.output.setDropout(rate[len(rate)-1])
	} else {
		nn.output.setDropout(rate[0])
	}
}

func (nn *DNN) Predict(inputs []float64) []float64 {
	for i := range nn.hidden {
		nn.hidden[i].forward(inputs, nn.activation)
		inputs = nn.hidden[i].outputs
	}
	nn.output.forward(inputs, nn.activation)

	return nn.output.outputs
}

// Train performs one iteration of prediction and backpropagation learning
func (nn *DNN) Train(inputs, targets []float64) {
	pred := nn.Predict(inputs)

	errors := make([]float64, len(nn.output.outputs))
	for i := range pred {
		errors[i] = nn.lossPrime(pred[i], targets[i]) * nn.variance(nn.output.outputs[i])
	}

	nn.output.backward(errors, nn.Rate)
	gradient, weight := errors, nn.output.weights
	for i := len(nn.hidden) - 1; i >= 0; i-- {
		errors = make([]float64, len(nn.hidden[i].outputs))
		for j := range weight {
			var sum float64
			for k := range gradient {
				sum += gradient[k] * weight[j][k]
			}
			errors[j] = sum * nn.variance(nn.hidden[i].outputs[j])
		}

		nn.hidden[i].backward(errors, nn.Rate)
		gradient, weight = errors, nn.hidden[i].weights
	}
}

func (nn *DNN) ValidateBinaryClassification(inputs, targets [][]float64) int {
	var truePred int

	for i, x := range inputs {
		var target int
		pred := nn.Predict(x)

		for j, y := range targets[i] {
			if y == 1.0 {
				target = j
				break
			}
		}

		if pred[target] == floats.Max(pred) {
			truePred++
		}
	}

	return truePred
}

type dnnLayer struct {
	inputs []float64

	weights [][]float64

	biases []float64

	outputs []float64

	dropout float64

	mask []bool
}

func newDropoutLayer(inputSize, outputSize int) *dnnLayer {
	layer := &dnnLayer{
		inputs:  make([]float64, inputSize),
		weights: make([][]float64, inputSize),
		biases:  make([]float64, outputSize),
		outputs: make([]float64, outputSize),
		dropout: 0.0,
		mask:    make([]bool, inputSize),
	}

	for i := range layer.weights {
		layer.weights[i] = make([]float64, outputSize)
	}
	layer.init()

	return layer
}

func (l *dnnLayer) init() {
	rand.Seed(time.Now().UnixNano())
	for i := range l.weights {
		for j := range l.weights[i] {
			l.weights[i][j] = rand.NormFloat64()
			l.biases[j] = rand.NormFloat64()
		}
	}
}

func (l *dnnLayer) setDropout(rate float64) {
	l.dropout = rate

	for i := range l.mask {
		l.mask[i] = rand.Float64() >= l.dropout
	}
}

func (l *dnnLayer) forward(inputs []float64, act ActivationFunc) {
	l.inputs = inputs

	for i := range l.outputs {
		sum := l.biases[i]
		for j := range l.inputs {
			if l.mask[j] {
				sum += l.inputs[j] * l.weights[j][i]
			}
		}
		l.outputs[i] = act(sum) / (1 - l.dropout)
	}
}

func (l *dnnLayer) backward(errors []float64, rate float64) {
	for i, x := range l.inputs {
		for j := range l.weights[i] {
			l.biases[j] -= errors[j] * x * rate
			if l.mask[i] {
				l.weights[i][j] -= errors[j] * x * rate
			}
		}
	}
}
