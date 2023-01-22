package gonn

import (
	"math/rand"
	"time"
)

type FFNN struct {
	Rate float64

	hiddenActivation, hiddenVariance ActivationFunc
	outputActivation, outputVariance ActivationFunc

	loss, lossPrime LossFunc

	hidden []*ffnnLayer

	output *ffnnLayer
}

func (nn *FFNN) Predict(inputs []float64) []float64 {
	for i := range nn.hidden {
		nn.hidden[i].forward(inputs, nn.hiddenActivation)
		inputs = nn.hidden[i].outputs
	}
	nn.output.forward(inputs, nn.outputActivation)

	return nn.output.outputs
}

type ffnnLayer struct {
	inputs []float64

	weights [][]float64

	biases []float64

	outputs []float64
}

func newFFNNLayer(inputSize, outputSize int) *ffnnLayer {
	layer := &ffnnLayer{
		inputs:  make([]float64, inputSize),
		weights: make([][]float64, inputSize),
		biases:  make([]float64, outputSize),
		outputs: make([]float64, outputSize),
	}

	rand.Seed(time.Now().UnixNano())
	for i := range layer.weights {
		layer.weights[i] = make([]float64, outputSize)
		for j := range layer.weights[i] {
			layer.weights[i][j] = rand.NormFloat64()
			layer.biases[j] = rand.NormFloat64()
		}
	}

	return layer
}

func (l *ffnnLayer) forward(inputs []float64, act ActivationFunc) {
	l.inputs = inputs

	for i := range l.outputs {
		sum := l.biases[i]
		for j := range l.inputs {
			sum += l.inputs[j] * l.weights[j][i]
		}
		l.outputs[i] = act(sum)
	}
}

func (l *ffnnLayer) backward(errors []float64, rate float64) {
	for i, x := range l.inputs {
		for j := range l.weights[i] {
			l.biases[j] -= errors[j] * x * rate
			l.weights[i][j] -= errors[j] * x * rate
		}
	}
}
