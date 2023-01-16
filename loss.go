package gonn

import "math"

type LossFunc func([]float64, []float64) float64

func MSE(pred, targets []float64) float64 {
	var mse float64

	for i := range pred {
		err := targets[i] - pred[i]
		mse += err * err
	}

	return mse / float64(len(pred))
}

func MAE(pred, targets []float64) float64 {
	var mae float64

	for i := range pred {
		err := pred[i] - targets[i]
		if err < 0 {
			mae -= err
		} else {
			mae += err
		}
	}

	return mae / float64(len(pred))
}

func Huber(pred, targets []float64) float64 {
	var sum float64
	delta := 1.0

	for i := range pred {
		err := targets[i] - pred[i]
		if math.Abs(err) > delta {
			sum += delta * (math.Abs(err) - 0.5*delta)
		} else {
			sum += 0.5 * err * err
		}
	}

	return sum
}
