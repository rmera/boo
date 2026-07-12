package utils

import (
	"math"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// puts in D the normalization output for the inputs in O.
// allocates a new matrix if D is nil.
func NormalizationDense(O, D *mat.Dense) *mat.Dense {
	return activationDense(O, D, Normalization)

}

// puts in probs the normalization output for the inputs in p.
// allocates a new slice if probs is nil.
func Normalization(p, probs []float64) []float64 {
	if probs == nil {
		probs = make([]float64, len(p))
	}
	sum := 0.0
	for i, v := range p {
		probs[i] = v
		sum += v
	}
	for i := range p {
		probs[i] /= sum
	}
	return probs
}

// puts in probs the softmax output for the inputs in p.
// allocates a new matrix if probs is nil.
func SoftMax(p, probs []float64) []float64 {
	if probs == nil {
		probs = make([]float64, len(p))
	}
	for i, v := range p {
		probs[i] = math.Exp(v)
	}
	den := floats.Sum(probs)
	for i := range p {
		probs[i] /= den
	}
	return probs
}

// puts in D the softmax output for the inputs in O.
// allocates a new matrix if D is nil.
func SoftMaxDense(O, D *mat.Dense) *mat.Dense {
	return activationDense(O, D, SoftMax)

}

// Applies the activation function given to each row of the O matrix to fill the D matrix
// and returns D. If nil is given for the D matrix, a new one is allocated.
func activationDense(O, D *mat.Dense, activation func([]float64, []float64) []float64) *mat.Dense {
	r, c := O.Dims()
	if D == nil {
		D = mat.NewDense(r, c, nil)
	}
	for i := 0; i < r; i++ {
		o := O.RawRowView(i)
		d := D.RawRowView(i)
		p := activation(o, d) //this should set D to the probabilities.
		D.SetRow(i, p)
	}
	return D
}

// IdentityDense is an 'activation function' that does nothing,
// but simply returns the O matrix. The D matrix is
// only required to comply with the activation function
// signature
func IdentityDense(O, D *mat.Dense) *mat.Dense {
	return O

}
