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
	Data        [][]float64 //each row is a feature vector, each col has a feature.
	Keys        []string
	Labels      []int
	FloatLabels []float64 //for now we keep both
	lower       bool      //signals if the Labels are in lower cap
}

// Sets all keys to lower cap
func (D *DataBunch) ToLower() {
	if D.lower {
		return
	}
	for i, v := range D.Keys {
		D.Keys[i] = strings.ToLower(v)
	}
	D.lower = true
}

// Returns a copy of the receiver. If you give blankdata and its true, you get
// a copy with correctly-sized but zeroed data (which is cheaper than copying the data
// if you will overwrite it anyway, as you'd do for bootstrapping
func (D *DataBunch) Copy(blankdata ...bool) *DataBunch {
	ret := new(DataBunch)
	ret.Keys = make([]string, len(D.Keys))
	copy(ret.Keys, D.Keys)
	ret.Labels = make([]int, len(D.Labels))
	copy(ret.Labels, D.Labels)
	ret.FloatLabels = make([]float64, len(D.FloatLabels))
	copy(ret.FloatLabels, D.FloatLabels)
	ret.Data = make([][]float64, len(D.Data))
	for i, v := range D.Data {
		ret.Data[i] = make([]float64, len(v))
		if len(blankdata) > 0 && blankdata[0] {
			continue
		}
		copy(ret.Data[i], v)
	}
	return ret
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
		dline := make([]string, 1, len(v)+1)
		dline[0] = fmt.Sprintf("%3d", D.getithLabel(i))
		for _, w := range v {
			s := fmt.Sprintf("%5.4f", w)
			dline = append(dline, s)

		}
		ret = append(ret, strings.Join(dline, " "))
	}
	return strings.Join(ret, "\n")

}

// I thought there would be something in the slices package to apply a function to all member of a slice
// but alas.
func tolower(s []string) []string {
	r := make([]string, len(s))
	for i, v := range s {
		r[i] = strings.ToLower(v)
	}
	return r
}

// should satisfy error
type featErr struct {
	nf []string
}

func (e *featErr) GetError() error {
	if e == nil || len(e.nf) == 0 {
		return nil
	}
	f := strings.Join(e.nf, ", ")
	return fmt.Errorf("Features " + f + " Not found in labels. Will exclude")

}

func newfeatErr() *featErr {
	r := new(featErr)
	r.nf = make([]string, 0, 2)
	return r
}

// Returns the IDs corresponding to the feature labels given. caps-sensitive if
// at least one caps is given and the first given is true. If one or more labels are
// not present it signals so in an error but it still returns the remaining IDs.
func (D *DataBunch) FeatIDsFromKeys(feats []string, nocaps ...bool) ([]int, error) {
	var err *featErr
	var nc bool
	l := D.Keys
	ret := make([]int, 0, len(feats))
	if len(nocaps) > 0 && nocaps[0] {
		nc = true
		l = tolower(D.Keys)
	}
	for _, v := range feats {
		if nc {
			v = strings.ToLower(v)
		}
		in := slices.Index(l, v)
		if in < 0 {
			if err == nil {
				err = newfeatErr()
			}
			err.nf = append(err.nf, v)
		} else {
			ret = append(ret, in)
		}
	}

	return ret, err.GetError()
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
		dline := make([]string, 1, l+1)
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
