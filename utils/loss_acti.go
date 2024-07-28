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

// Interface for loss functions used in the package.
type LossFunc interface {
	Loss(*mat.Dense, *mat.Dense, *mat.Dense) float64
	Name() string
	NegGradients(*mat.Dense, *mat.Dense, *mat.Dense) *mat.Dense
	Gradients(*mat.Dense, *mat.Dense, *mat.Dense) *mat.Dense
	Hessian(*mat.Dense, *mat.Dense) *mat.Dense
}

// Square error
type SQErrLoss struct {
}

func (sq *SQErrLoss) Name() string { return "sqerr" }

func (sq *SQErrLoss) Loss(y, pred, loss *mat.Dense) float64 {
	r, c := y.Dims()
	if loss == nil {
		loss = mat.NewDense(r, c, nil)
	}
	loss.Sub(y, pred)
	loss.MulElem(loss, loss)
	m := loss.RawMatrix().Data
	l := 0.0
	n := 0
	for _, v := range m {
		l += v
		n++
	}
	return l / float64(n)

}

// Returns the results matrix filled witht he negative gradients.
// if a nil results is given, it will allocate a new matrix and return it.
func (m *SQErrLoss) NegGradients(yohelabels, pred, results *mat.Dense) *mat.Dense {
	if results == nil {
		r, c := yohelabels.Dims()
		results = mat.NewDense(r, c, nil)
	}
	results.Sub(yohelabels, pred)
	return results
}

// Returns the results matrix filled with the gradients.
// if a nil results is given, it will allocate a new matrix and return it.
func (m *SQErrLoss) Gradients(yohelabels, pred, results *mat.Dense) *mat.Dense {
	g := m.NegGradients(yohelabels, pred, results)
	g.Scale(-1, g)
	return g

}

// Returns the results matrix filled with the Hessian.
// if a nil results is given, it will allocate a new matrix and return it.
func (m *SQErrLoss) Hessian(probabilities, hessian *mat.Dense) *mat.Dense {
	r, c := probabilities.Dims()
	if hessian == nil {
		hessian = mat.NewDense(r, c, nil)
	}
	//so, a "plain" hessian.
	ToOnes(hessian)
	return hessian
}

// the error used in regular gradient boosting
type MSELoss struct {
}

func (mse *MSELoss) Loss(y, pred, loss *mat.Dense) float64 {
	r, c := y.Dims()
	if loss == nil {
		loss = mat.NewDense(r, c, nil)
	}
	loss.Sub(y, pred)
	loss.MulElem(loss, loss)
	m := loss.RawMatrix().Data
	l := 0.0
	n := 0
	for _, v := range m {
		l += v
		n++
	}
	return l / float64(n)

}

func (m *MSELoss) Name() string {
	return "mse"
}

// Returns the results matrix filled witht he negative gradients.
// if a nil results is given, it will allocate a new matrix and return it.
func (m *MSELoss) NegGradients(yohelabels, probabilities, results *mat.Dense) *mat.Dense {
	if results == nil {
		r, c := yohelabels.Dims()
		results = mat.NewDense(r, c, nil)
	}
	results.Sub(yohelabels, probabilities)
	return results
}

// Returns the results matrix filled with the gradients.
// if a nil results is given, it will allocate a new matrix and return it.
func (m *MSELoss) Gradients(yohelabels, probabilities, results *mat.Dense) *mat.Dense {
	g := m.NegGradients(yohelabels, probabilities, results)
	g.Scale(-1, g)
	return g

}

// Returns the results matrix filled with the Hessian.
// if a nil results is given, it will allocate a new matrix and return it.
func (m *MSELoss) Hessian(probabilities, hessian *mat.Dense) *mat.Dense {
	r, c := probabilities.Dims()
	if hessian == nil {
		hessian = mat.NewDense(r, c, nil)
	}

	probraw := probabilities.RawMatrix().Data
	hraw := hessian.RawMatrix().Data
	for i, v := range probraw {
		hraw[i] = (1 - v) * v
	}
	return hessian
}
