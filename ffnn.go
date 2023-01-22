package gonn

import (
	"gonum.org/v1/gonum/floats"
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

func (nn *FFNN) Train(inputs, targets []float64) {
	pred := nn.Predict(inputs)

	errors := make([]float64, len(nn.output.outputs))
	for i, p := range pred {
		errors[i] = nn.lossPrime(p, targets[i]) * nn.outputVariance(p)
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
			errors[j] = sum * nn.hiddenVariance(nn.hidden[i].outputs[j])
		}
		nn.hidden[i].backward(errors, nn.Rate)
		gradient, weight = errors, nn.hidden[i].weights
	}
}

func (nn *FFNN) ValidateBinaryClassification(inputs, targets [][]float64) int {
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
