package gonn

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
	"strings"
)

// NewDNN creates a new Dropout Multilayer Perceptron
//
// sizes contains the sizes of each layer in the network
//
// learningRate is the scaling hyperparameter for gradient descent
//
// activation defines the normalization function applied to the output
//
// variance defines the derivative of the activation function used
func NewDNN(sizes []int, learningRate float64, hiddenActivation, hiddenVariance, outputActivation, outputVariance ActivationFunc, loss, lossPrime LossFunc) *DNN {
	if len(sizes) < 2 {
		panic("invalid slices parameter supplied to NewDNN")
	}

	layers := make([]*dnnLayer, len(sizes)-2)
	for i := 1; i <= len(layers); i++ {
		layers[i-1] = newDropoutLayer(sizes[i-1], sizes[i])
	}

	return &DNN{
		Rate:             learningRate,
		hiddenActivation: hiddenActivation,
		hiddenVariance:   hiddenVariance,
		outputActivation: outputActivation,
		outputVariance:   outputVariance,
		loss:             loss,
		lossPrime:        lossPrime,
		hidden:           layers,
		output:           newDropoutLayer(sizes[len(sizes)-2], sizes[len(sizes)-1]),
	}
}

// LoadCSVData loads training data from source and returns two float64 matrices containing the training data and the target outcome.
//
// inputLen determines how many fields contain the input data.
//
// outputLen determines the number of targets.
//
// names signifies if the first entry in the CSV contains field names.
func LoadCSVData(src string, inputLen, targetLen int, names bool) ([][]float64, [][]float64) {
	entryLen := inputLen + targetLen
	f, err := os.Open(src)
	if err != nil {
		log.Fatal(err.Error())
	}
	defer f.Close()

	rdr := csv.NewReader(f)
	rdr.FieldsPerRecord = entryLen
	data, err := rdr.ReadAll()
	if err != nil {
		log.Fatal(err.Error())
	}

	if names {
		data = data[1:]
	}
	inputs, targets := make([][]float64, len(data)), make([][]float64, len(data))

	for k, x := range data {
		inputs[k], targets[k] = make([]float64, inputLen), make([]float64, targetLen)
		i, j := 0, 0

		for n, y := range x {
			y = strings.Replace(y, ",", "", -1)
			d, err := strconv.ParseFloat(y, 64)
			if err != nil {
				log.Fatal(err.Error())
			}

			if n < inputLen {
				inputs[k][i] = d
				i++
				continue
			}
			targets[k][j] = d
			j++
		}
	}

	return inputs, targets
}
