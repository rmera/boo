package utils

import (
	"fmt"
	"slices"
	"strings"

	"gonum.org/v1/gonum/mat"
)

// A simple structure for data
// Keys are the feature names, Lables are the
// classification of each Data vector, if available
type DataBunch struct {
	Data        [][]float64
	Keys        []string
	Labels      []int
	FloatLabels []float64 //for now we keep both
}

// returns a one-hot-encoded representation of the keys of the data bunch
func (D *DataBunch) OHEKeys() (*mat.Dense, []string) {
	return oneHotEncodeDense(D.Keys)
}

// Returns a one-hot-encoded representation of the labels of the data bunch.
func (D *DataBunch) OHELabels() (*mat.Dense, []int) {
	return oneHotEncodeDense(D.Labels)
}

func (D *DataBunch) getithLabel(i int) int {
	if len(D.Labels) == len(D.Data) {
		return D.Labels[i]
	} else {
		return -1 //the bunch has no labels
	}
}

// Returns a string representation of the data bunch
func (D *DataBunch) String() string {
	if D == nil {
		return ""
	}
	ret := make([]string, 0, 1+len(D.Data))
	ret = append(ret, "Labels "+strings.Join(D.Keys, " "))
	for i, v := range D.Data {
		dline := make([]string, len(v)+1)
		dline[0] = fmt.Sprintf("%3d", D.getithLabel(i))
		for _, w := range v {
			s := fmt.Sprintf("%5.4f", w)
			dline = append(dline, s)

		}
		ret = append(ret, strings.Join(dline, " "))
	}
	return strings.Join(ret, "\n")

}

// returns the data in libSVM format
// not very good as it doesn't omit the zero-valued data, but I'll get there.
func (D *DataBunch) LibSVM() string {
	if D == nil {
		return ""
	}
	ret := make([]string, 0, len(D.Data)+1)
	l := len(D.Data[0])   //everything should be this lenght
	if len(D.Keys) == l { //I asume no keys otherwise
		k := make([]string, 1, l+1)
		k[0] = "Labels"
		for i, v := range D.Keys {
			s := fmt.Sprintf("%d:%s", i+1, v)
			k = append(k, s)
		}
		ret = append(ret, strings.Join(k, " "))

	}
	for i, v := range D.Data {
		dline := make([]string, l+1)
		dline[0] = fmt.Sprintf("%3d", D.getithLabel(i))
		for j, w := range v {
			s := fmt.Sprintf("%d:%5.4f", j+1, w)
			dline = append(dline, s)

		}
		ret = append(ret, strings.Join(dline, " "))

	}
	return strings.Join(ret, "\n")

}

// Returns a one-hot-encoded representation of the labels of the data bunch.
func (D *DataBunch) LabelsRegression() *mat.Dense {
	//cols :=1
	//	datapoints := len(labels)
	//rows := datapoints
	ol := make([]float64, len(D.FloatLabels))
	//	fmt.Println(D.FloatLabels) /////
	copy(ol, D.FloatLabels)
	ohlabels := mat.NewDense(len(D.FloatLabels), 1, ol)
	return ohlabels

}

// each row is a feature vector, cols are the features
func oneHotEncodeDense[S ~[]E, E Encodeable](labels S) (*mat.Dense, S) {
	de := distinctElements(labels)
	cols := len(de)
	datapoints := len(labels)
	rows := datapoints
	ohlabels := mat.NewDense(rows, cols, make([]float64, rows*cols))
	for i, v := range labels {
		index := slices.Index(de, v)
		ohlabels.Set(i, index, 1.0)
	}
	return ohlabels, de

}
