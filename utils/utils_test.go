package utils

import (
	"fmt"
	"math"
	"slices"
	"sort"
	"testing"

	"gonum.org/v1/gonum/mat"
)

func TestCSV(Te *testing.T) {
	data, err := DataBunchFromCSVFile("../tests/train.csv", true, true)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println(data.String())
	fmt.Println(data.CSV())
	data, err = DataBunchFromCSVFile("../tests/trainnoheader.csv", false, true)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("no header", data.String())
	data, err = DataBunchFromCSVFile("../tests/trainnolabels.csv", true, false)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("no labels", data.String())
	data, err = DataBunchFromCSVFile("../tests/train.csv", true, true)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("bad read", data.String())

}

func TestLibSVM(Te *testing.T) {
	fmt.Println("To read traineasy")
	data, err := DataBunchFromLibSVMFile("../tests/traineasy.svm", true)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println(data.String())
	fmt.Println(data.LibSVM())
	fmt.Println("Toread traineasynoheader")
	data, err = DataBunchFromLibSVMFile("../tests/traineasynoheader.svm", false)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("no headers", data.String())
	fmt.Println("To read trainneasynolabels")
	data, err = DataBunchFromLibSVMFile("../tests/traineasynolabels.svm", true)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("no labels", data.String())

}

func TestSampleSlice(Te *testing.T) {
	test := []float64{0, 1, 2, 3, 4, 5, 6, 7, 8, 9}
	indx := []int{0, 3, 5, 9}
	nt := SampleSlice(test, indx)
	if !slices.Equal([]float64{0, 3, 5, 9}, nt) {
		Te.Error("Problem with SampleSlice")
	}
	fmt.Println(test, nt, indx)
}

func TestMatrixFunctions(Te *testing.T) {
	d := mat.NewDense(3, 3, []float64{1, 2, 3, 4, 5, 6, 7, 8, 9})
	col := DenseCol(d, 1)
	fmt.Printf("%v, %v\n", d, col)
	if !slices.Equal(col.RawRowView(0), []float64{2, 5, 8}) {
		Te.Error("Problem with DenseCol")
	}
	AddToCol(d, []float64{1, 2, 1}, 1)
	fmt.Println(PrintDenseMatrix(d))
	fm := [][]float64{{1, 2, 3}, {4, 5, 6}, {7, 8, 9}}
	tfm := TransposeFloats(fm)
	fmt.Println(tfm)
	fm2 := [][]float64{{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}}
	sfm2 := SampleMatrix(fm2, []int{0, 1}, []int{1, 2})
	fmt.Println(sfm2)

}

func TestArgSort(Te *testing.T) {
	tosort := []float64{9, 8, 7, 6, 5, 4, 3, 2, 1}
	i, f := ArgSort(tosort)
	fmt.Println(i, f)

}

/*********** Activation testss *********/

func TestNormalizationFloats(Te *testing.T) {
	p := []float64{1, 2, 3, 4}
	got := NormalizationFloats(p, nil)
	want := []float64{0.1, 0.2, 0.3, 0.4}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			Te.Errorf("NormalizationFloats: got %v, want %v", got, want)
			break
		}
	}
	// a non-nil dst should be reused, not replaced
	dst := make([]float64, 4)
	got2 := NormalizationFloats(p, dst)
	if &got2[0] != &dst[0] {
		Te.Error("NormalizationFloats: didn't reuse the provided destination slice")
	}
}

func TestSoftMaxFloats(Te *testing.T) {
	p := []float64{0, 0, 0}
	got := SoftMaxFloats(p, nil)
	want := []float64{1.0 / 3, 1.0 / 3, 1.0 / 3}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-9 {
			Te.Errorf("SoftMaxFloats: got %v, want %v", got, want)
		}
	}
	sum := 0.0
	for _, v := range got {
		sum += v
	}
	if math.Abs(sum-1) > 1e-9 {
		Te.Errorf("SoftMaxFloats: probabilities don't sum to 1: %v", sum)
	}

	// a larger input for one class should dominate the probability mass
	p2 := []float64{0, 10}
	got2 := SoftMaxFloats(p2, nil)
	if got2[1] <= got2[0] {
		Te.Errorf("SoftMaxFloats: expected class with larger logit to dominate, got %v", got2)
	}
}

func TestActivationTypes(Te *testing.T) {
	in := mat.NewDense(2, 3, []float64{1, 2, 3, 4, 5, 6})

	norm := &Normalization{}
	if norm.Name() != "normalization" {
		Te.Errorf("Normalization.Name(): got %q, want %q", norm.Name(), "normalization")
	}
	nout := norm.Acti(in, nil)
	checkRowsSumToOne(Te, "Normalization.Acti", nout)

	sm := &SoftMax{}
	if sm.Name() != "softmax" {
		Te.Errorf("SoftMax.Name(): got %q, want %q", sm.Name(), "softmax")
	}
	sout := sm.Acti(in, nil)
	checkRowsSumToOne(Te, "SoftMax.Acti", sout)

	id := &Identity{}
	if id.Name() != "identity" {
		Te.Errorf("Identity.Name(): got %q, want %q", id.Name(), "identity")
	}
	iout := id.Acti(in, nil)
	if iout != in {
		Te.Error("Identity.Acti: should return the input matrix unchanged")
	}
}

func checkRowsSumToOne(Te *testing.T, name string, m *mat.Dense) {
	Te.Helper()
	r, c := m.Dims()
	for i := range r {
		sum := 0.0
		for j := range c {
			sum += m.At(i, j)
		}
		if math.Abs(sum-1) > 1e-9 {
			Te.Errorf("%s: row %d doesn't sum to 1: %v", name, i, sum)
		}
	}
}

func TestSlice(Te *testing.T) {
	s := NewSlice([]float64{3, 1, 2})
	wantIdx := []int{0, 1, 2}
	if !slices.Equal(s.Idx, wantIdx) {
		Te.Errorf("NewSlice: initial Idx got %v, want %v", s.Idx, wantIdx)
	}
	sort.Sort(s)
	wantVals := []float64{1, 2, 3}
	if !slices.Equal([]float64(s.Float64Slice), wantVals) {
		Te.Errorf("Sort: values got %v, want %v", []float64(s.Float64Slice), wantVals)
	}
	// Idx must track where each sorted value came from originally
	wantSortedIdx := []int{1, 2, 0}
	if !slices.Equal(s.Idx, wantSortedIdx) {
		Te.Errorf("Sort: Idx got %v, want %v", s.Idx, wantSortedIdx)
	}
}
