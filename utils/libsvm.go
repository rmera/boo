package utils

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
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

// This is not very good at all, it ignores the main strenght of libSVM format, that you can omit 0 values.
// It's just a temporary solution to allow some testing
func parseLibSVMLine(line string, header bool, retstr []string, retflo []float64) (int, float64, []string, []float64, error) {
	var rets []string
	var retf []float64
	var class int
	var fclass float64
	var err error
	fields := strings.Fields(line)
	if len(retstr) >= len(fields)-1 {
		rets = retstr[:0]
	}
	if len(retflo) >= len(fields)-1 {
		retf = retflo[:]
	}

	labeloffset := 1
	if strings.Contains(fields[0], ":") {
		//This condition means that the first field is already a feature field (has a ":"
		//meaning that there is no label in this line.
		labeloffset = 0
	}
	if !header && labeloffset == 1 { //we don't do this if we don't have labels
		class, err = strconv.Atoi(fields[0])
		if err != nil {
			fclass, err = strconv.ParseFloat(fields[0], 64)
			if err != nil {
				return 0, 0, nil, nil, err
			}
		}
	}
	for _, v := range fields[labeloffset:] {
		feats := strings.Split(v, ":")
		if len(feats) != 2 {
			return -1, -1, nil, nil, fmt.Errorf("Malformed term: %s", v)
		}
		feat := feats[1]
		if header {
			rets = append(rets, feat)
		} else {
			val, err := strconv.ParseFloat(feat, 64)
			if err != nil {
				return class, -1, nil, nil, err
			}
			retf = append(retf, val)
		}

	}
	return class, fclass, rets, retf, nil
}

func svmliberror(err error, linenu int, line string) error {
	return fmt.Errorf("Can't read line %d in libSVM-formatted file, Error: %v, line: %s", linenu, err, line)
}

// reads a libSVM-formatted file and returns a DataBunch. It's a pretty poor reader right now
// as it doesn't support sparse-libSVM files. The file has to have missing data points explitly (say, set to 0)
// and it needs to have labels.
func DataBunchFromLibSVMFile(filename string, hasHeader ...bool) (*DataBunch, error) {
	hasaheader := false
	if len(hasHeader) > 0 {
		hasaheader = hasHeader[0]
	}

	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	return ParseLibSVMFromReader(f, hasaheader)
}

func ParseLibSVMFromReader(r io.Reader, hasHeader bool) (*DataBunch, error) {
	buf := bufio.NewReader(r)
	var line string
	var err2 error
	var headers []string
	var data [][]float64 = make([][]float64, 0, 1)
	var labels []int
	var flabels []float64
	cont := 0
	for {
		//	println("Will read the line", cont+1) ///////
		line, err2 = buf.ReadString('\n')
		if err2 != nil {
			break
		}
		var err error
		if hasHeader && cont == 0 {
			_, _, headers, _, err = parseLibSVMLine(line, true, nil, nil)
			if err != nil {
				return nil, svmliberror(err, cont+1, line)
			}
			cont++
			continue
		}
		t := make([]float64, 0, len(headers))
		l := 0
		fl := 0.0
		l, fl, _, t, err = parseLibSVMLine(line, false, nil, nil)
		if err != nil {
			return nil, svmliberror(err, cont+1, line)

		}
		data = append(data, t)
		labels = append(labels, l)
		flabels = append(flabels, fl)

		cont++
	}
	if err2.Error() != "EOF" {
		return nil, err2
	}

	return &DataBunch{Data: data, Labels: labels, Keys: headers, FloatLabels: flabels}, nil
}

/*
// This should work, as I pretty much just use golearn's machinery.
func DataBunchFromLibCSVFile(filename string, hasHeader bool) (*DataBunch, error) {
	instanceql, err := ParseCSVToInstances(filename, hasHeader)
	if err != nil {
		return nil, err
	}
	ret := &DataBunch{}
	ret.Data = convertInstancesToProblemVec(instanceql)
	ret.FloatLabels, err = convertInstancesToLabelVec(instanceql)
	if err != nil {
		return nil, err
	}
	for _, v := range ret.FloatLabels {
		il := int(math.Round(v))
		ret.Labels = append(ret.Labels, il)
	}
	return ret, nil
}
**/

type Encodeable interface {
	~int | ~int8 | ~int16 | ~int32 | ~int64 |
		~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64 |
		~string
}

func distinctElements[S ~[]E, E Encodeable](s S) S {
	diff := make(S, 0, len(s)/2)
	for _, v := range s {
		if !slices.Contains(diff, v) {
			diff = append(diff, v)
		}
	}
	return diff
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
