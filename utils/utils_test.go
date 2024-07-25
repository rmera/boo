package utils

import (
	"fmt"
	"slices"
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
	data, err = DataBunchFromCSVFile("../tests/train.csv", false, true)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("bad read", data.String())

}

func TestLibSVM(Te *testing.T) {
	data, err := DataBunchFromLibSVMFile("../tests/traineasy.svm", true)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println(data.String())
	fmt.Println(data.LibSVM())
	data, err = DataBunchFromLibSVMFile("../tests/traineasynoheader.svm", false)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("no headers", data.String())
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
