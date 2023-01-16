package gonn

import "math"

type ActivationFunc func(float64) float64

func NoActivation(x float64) float64 {
	return x
}

func Sigmoid(x float64) float64 {
	return 1 / (1 + math.Exp(-x))
}

func SigmoidPrime(x float64) float64 {
	return Sigmoid(x) * (1 - Sigmoid(x))
}

func Tanh(x float64) float64 {
	return 2*Sigmoid(2*x) - 1
}

func TanhPrime(x float64) float64 {
	return 2 * SigmoidPrime(2*x)
}

func Softplus(x float64) float64 {
	return math.Log(1 + math.Exp(x))
}

func SoftplusPrime(x float64) float64 {
	return Sigmoid(x)
}

func ReLU(x float64) float64 {
	return math.Max(0, x)
}

func LeakyReLU(x float64) float64 {
	if x > 0 {
		return x
	}
	return 0.01 * x
}

func ReLUPrime(x float64) float64 {
	if x > 0 {
		return 1
	}
	return 0
}
