package learn

import (
	"fmt"
	"math"
	"sort"
	"strings"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

//func UpdateCols(m *mat.Dense, s []float64) {

//}

//utils

// takes a sub-matrix of m, consisting in its rows with indexes present in rowIndexes and the
// columns present in colIndexes, inthe order in which they appear in the 2 int slices.
func SampleMatrix(m [][]float64, rowIndexes, colIndexes []int) [][]float64 {
	ret := make([][]float64, 0, len(rowIndexes))
	for _, v := range rowIndexes {
		s := SampleSlice(m[v], colIndexes)
		ret = append(ret, s)
	}
	return ret
}

// Produces a slice that contains each ith element (zero-based) in the slice s, where i is an element in indexes
// in the order in which they appear in indexes.
func SampleSlice(s []float64, indexes []int) []float64 {
	if len(indexes) > len(s) {
		panic(fmt.Sprintf("SampleSlice: can't sample %d values from slice of len %d: %v %v", len(indexes), len(s), indexes, s))
	}
	ret := make([]float64, 0, len(indexes))
	for _, v := range indexes {
		if v >= len(s) {
			panic(fmt.Sprintf("SampleSlice: can't sample value %d from slice of len %d: %v", v, len(s), s))
		}
		ret = append(ret, s[v])
	}
	return ret
}

type idSorter struct {
	i []int
	a []float64
}

func (s *idSorter) Len() int           { return len(s.i) }
func (s *idSorter) Less(i, j int) bool { return s.a[i] < s.a[j] }
func (s *idSorter) Swap(i, j int) {
	s.a[i], s.a[j] = s.a[j], s.a[i]
	s.i[i], s.i[j] = s.i[j], s.i[i]
}

// Returns the indexes that would sort the given slice
// and also the sorted slice. It doesn't touch the original slice!
func ArgSort(a []float64) ([]int, []float64) {
	in := make([]int, 0, len(a))
	val := make([]float64, 0, len(a))
	for i, v := range a {
		in = append(in, i)
		val = append(val, v)
	}
	r := &idSorter{i: in, a: val}
	sort.Sort(r)
	return r.i, r.a
}

// Transposes the matrix represented by the [][]m slice of slices.
func TransposeFloats(m [][]float64) [][]float64 {
	ret := make([][]float64, 0, len(m[0]))
	for i := range m[0] {
		r := make([]float64, 0, len(m))
		for j := range m {
			r = append(r, m[j][i])
		}
		ret = append(ret, r)
	}
	return ret
}

// Produces a printable version of m
func PrintDenseMatrix(m *mat.Dense) string {
	r, _ := m.Dims()
	ret := make([]string, 0, r)
	for i := 0; i < r; i++ {
		ret = append(ret, fmt.Sprintf("%v", m.RawRowView(i)))
	}
	ret[0] = "[" + ret[0]
	ret[len(ret)-1] = ret[len(ret)-1] + "]"
	return strings.Join(ret, "\n")
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
	r, c := O.Dims()
	if D == nil {
		D = mat.NewDense(r, c, nil)
	}
	for i := 0; i < r; i++ {
		o := O.RawRowView(i)
		d := D.RawRowView(i)
		p := SoftMax(o, d) //this should set D to the probabilities.
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

// Returns the col-th column of the D matrix as a row Dense matrix
func DenseCol(D *mat.Dense, col int) *mat.Dense {
	raw := D.RawMatrix()
	r, c := D.Dims()
	if col > c {
		panic(fmt.Sprintf("DenseCol: Requested column %d of matrix with %d columns: %v", col, c, D))
	}
	data := raw.Data
	newcol := make([]float64, 0, r)
	for i := col; i < len(data); i += c {
		newcol = append(newcol, data[i])
	}
	return mat.NewDense(1, len(newcol), newcol)
}

// adds the i-th number in s to the ith row to the col column of
// D. s must have as many elements as D has rows.
func AddToCol(D *mat.Dense, s []float64, col int) {
	raw := D.RawMatrix()
	r, c := D.Dims()
	if col > c {
		panic(fmt.Sprintf("AddToCol: Requested column %d of matrix with %d columns: %v", col, c, D))
	}
	if len(s) != r {

		panic(fmt.Sprintf("AddToCol: Tried to add a %d-elements slice to a %d rows column %v %v", len(s), r, s, D))
	}
	data := raw.Data
	cont := 0
	for i := col; i < len(data); i += c {
		data[i] += s[cont]
		cont++
	}
	return
}

// Sets all entries of m to 1.0
func ToOnes(m *mat.Dense) {
	rm := m.RawMatrix().Data
	for i := range rm {
		rm[i] = 1.0
	}
}
