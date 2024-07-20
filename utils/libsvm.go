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

func (D *DataBunch) OHEKeys() (*mat.Dense, []string) {
	return oneHotEncodeDense(D.Keys)
}

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
func parseLibSVMLine(line string, header bool, retstr []string, retflo []float64) (int, []string, []float64, error) {
	var rets []string
	var retf []float64
	var class int
	var err error
	fields := strings.Fields(line)
	if len(retstr) >= len(fields)-1 {
		rets = retstr[:0]
	}
	if len(retflo) >= len(fields)-1 {
		retf = retflo[:]
	}
	if !header {
		class, err = strconv.Atoi(fields[0])
		if err != nil {
			return 0, nil, nil, err
		}
	}
	for _, v := range fields[1:] {
		feats := strings.Split(v, ":")
		if len(feats) != 2 {
			return -1, nil, nil, fmt.Errorf("Malformed term: %s", v)
		}
		feat := feats[1]
		if header {
			rets = append(rets, feat)
		} else {
			val, err := strconv.ParseFloat(feat, 64)
			if err != nil {
				return class, nil, nil, err
			}
			retf = append(retf, val)
		}

	}
	return class, rets, retf, nil
}

func svmliberror(err error, linenu int, line string) error {
	return fmt.Errorf("Can't read line %d in libSVM-formatted file, Error: %v, line: %s", linenu, err, line)
}

func DataBunchFromLibSVMFile(filename string, hasHeader bool) (*DataBunch, error) {
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	return ParseLibSVMFromReader(f, hasHeader)
}

func ParseLibSVMFromReader(r io.Reader, hasHeader bool) (*DataBunch, error) {
	buf := bufio.NewReader(r)
	var line string
	var err2 error
	var headers []string
	var data [][]float64 = make([][]float64, 0, 1)
	var labels []int
	cont := 0
	for {
		//	println("Will read the line", cont+1) ///////
		line, err2 = buf.ReadString('\n')
		if err2 != nil {
			break
		}
		var err error
		if hasHeader && cont == 0 {
			_, headers, _, err = parseLibSVMLine(line, true, nil, nil)
			if err != nil {
				return nil, svmliberror(err, cont+1, line)
			}
			cont++
			continue
		}
		t := make([]float64, 0, len(headers))
		l := 0
		l, _, t, err = parseLibSVMLine(line, false, nil, nil)
		if err != nil {
			return nil, svmliberror(err, cont+1, line)

		}
		data = append(data, t)
		labels = append(labels, l)

		cont++
	}
	if err2.Error() != "EOF" {
		return nil, err2
	}

	return &DataBunch{Data: data, Labels: labels, Keys: headers}, nil
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
