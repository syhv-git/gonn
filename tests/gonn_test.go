package tests

import (
	gonn "github.com/syhv-git/gonn"
	"log"
	"testing"
)

func TestDNN(t *testing.T) {
	sizes, drop, epochs := []int{4, 3, 4, 3}, []float64{0.4}, 500

	dnn := gonn.NewDNN(sizes, 0.4, gonn.Sigmoid, gonn.SigmoidPrime)

	src := "training_data.csv"
	inputs, targets := gonn.LoadCSVData(src, 4, 3, true)

	for i := 0; i < epochs; i++ {
		dnn.SetDropout(drop)
		for j := range inputs {
			dnn.Train(inputs[j], targets[j])
		}
	}

	test := "test_data.csv"
	in, tar := gonn.LoadCSVData(test, 4, 3, true)

	dnn.SetDropout([]float64{0})
	pred := dnn.Test(in, tar)

	log.Printf("\nExpected: %d\nPredicted: %d", len(in), pred)
	pred2 := dnn.Test(inputs, targets)

	log.Printf("\nExpected: %d\nPredicted: %d", len(inputs), pred2)
}
