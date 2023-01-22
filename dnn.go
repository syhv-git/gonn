package gonn

import (
	"gonum.org/v1/gonum/floats"
	"log"
	"math/rand"
)

type DNN struct {
	Rate float64

	hiddenActivation, hiddenVariance ActivationFunc
	outputActivation, outputVariance ActivationFunc

	loss, lossPrime LossFunc

	hidden []*dnnLayer

	output *dnnLayer
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
		nn.hidden[i].forward(inputs, nn.hiddenActivation)
		inputs = nn.hidden[i].outputs
	}
	nn.output.forward(inputs, nn.outputActivation)

	return nn.output.outputs
}

// Train performs one iteration of prediction and backpropagation learning
func (nn *DNN) Train(inputs, targets []float64) {
	pred := nn.Predict(inputs)

	errors := make([]float64, len(nn.output.outputs))
	for i, p := range pred {
		errors[i] = nn.lossPrime(p, targets[i]) * nn.outputVariance(p)
	}

	nn.output.backward(errors, nn.Rate)
	gradient, weight, mask, drop := errors, nn.output.weights, nn.output.mask, nn.output.dropout
	for i := len(nn.hidden) - 1; i >= 0; i-- {
		errors = make([]float64, len(nn.hidden[i].outputs))
		for j := range weight {
			if mask[j] {
				var sum float64
				for k := range gradient {
					sum += gradient[k] * weight[j][k]
				}
				errors[j] = sum * nn.hiddenVariance(nn.hidden[i].outputs[j]) / (1 - drop)
			} else {
				errors[j] = 0
			}
		}

		nn.hidden[i].backward(errors, nn.Rate)
		gradient, weight, mask, drop = errors, nn.hidden[i].weights, nn.hidden[i].mask, nn.hidden[i].dropout
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
	*ffnnLayer

	dropout float64

	mask []bool
}

func newDropoutLayer(inputSize, outputSize int) *dnnLayer {
	return &dnnLayer{
		ffnnLayer: newFFNNLayer(inputSize, outputSize),
		dropout:   0.0,
		mask:      make([]bool, inputSize),
	}
}

func (l *dnnLayer) setDropout(rate float64) {
	l.dropout = rate
	c := 0
	for i := range l.mask {
		l.mask[i] = rand.Float64() >= l.dropout
		if l.mask[i] {
			c++
		}
	}
	if c == 0 {
		l.mask[0] = true
	}
}

func (l *dnnLayer) forward(inputs []float64, act ActivationFunc) {
	l.inputs = inputs

	for i := range l.outputs {
		sum := l.biases[i]
		for j := range l.inputs {
			if l.mask[j] {
				sum += l.inputs[j] * l.weights[j][i] / (1 - l.dropout)
			}
		}
		l.outputs[i] = act(sum)
	}
}

func (l *dnnLayer) backward(errors []float64, rate float64) {
	for i, x := range l.inputs {
		for j := range l.weights[i] {
			if l.mask[i] {
				l.biases[j] -= errors[j] * rate
				l.weights[i][j] -= errors[j] * x * rate
			}
		}
	}
}
