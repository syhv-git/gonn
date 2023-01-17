package gonn

import "math"

type LossFunc func(float64, float64) float64

func MSE(predicted, target float64) float64 {
	err := target - predicted
	mse := err * err / 2
	return mse
}

func MSEPrime(predicted, target float64) float64 {
	return target - predicted
}

func MAE(predicted, target float64) float64 {
	return math.Abs(target - predicted)
}

func MAEPrime(predicted, target float64) float64 {
	err := target - predicted
	if err == 0 {
		return 0
	}

	return err / math.Abs(err)
}
