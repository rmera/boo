package utils

import (
	"gonum.org/v1/gonum/mat"
)

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
