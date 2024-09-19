package utils

import (
	"fmt"
	"sort"
	"strings"

	"gonum.org/v1/gonum/mat"
)

//utils

// takes a sub-matrix of m, consisting in its rows with indexes present in rowIndexes and the
// columns present in colIndexes, inthe order in which they appear in the 2 int slices.
func SampleColAndTranspose(m [][]float64, target []float64, targetindexes, rowIndexes []int, colIndex int) ([]float64, []int) {

	size := len(rowIndexes)
	if len(target) < size || len(targetindexes) < size {
		p := fmt.Sprintf("SamplecolAndTranspose: the target (%d) or targetindexes (%d) is too small to store the sampled values of the matrix %d", len(target), len(targetindexes), size)
		panic(p)
	}
	if len(target) > size {
		target = target[:size]
	}
	if len(targetindexes) > size {
		targetindexes = targetindexes[:size]
	}

	for i, v := range rowIndexes {
		target[i] = m[v][colIndex]
		targetindexes[i] = i
	}
	return target, targetindexes
}

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
func MemArgSort(target []float64, tmpint []int, tmpval []float64) ([]int, []float64) {
	for i, v := range target {
		tmpint[i] = i // = append(in, i)
		tmpval[i] = v // = append(val, v)
	}
	r := &idSorter{i: tmpint, a: tmpval}
	sort.Sort(r)
	return r.i, r.a
}

// Returns the indexes that would sort the given slice
// and also the sorted slice, which is the same given
// (i.e. it gets overwritten)
func OWArgSort(a []float64, in []int) ([]int, []float64) {
	r := &idSorter{i: in, a: a}
	sort.Sort(r)
	return r.i, r.a
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
