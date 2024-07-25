package utils

import (
	"encoding/csv"
	"errors"
	"fmt"
	"io"
	"os"
	"strconv"
	"strings"
)

// returns the data in CSV format, using whatever string is given as a separator
// (comma if no string is given)
func (D *DataBunch) CSV(separator ...string) string {
	if D == nil {
		return ""
	}
	sep := ","
	if len(separator) > 0 {
		sep = separator[0]
	}
	ret := make([]string, 0, len(D.Data)+1)
	l := len(D.Data[0])   //everything should be this lenght
	if len(D.Keys) == l { //I asume no keys otherwise
		k := make([]string, 1, l+1)
		k[0] = "Labels"
		for _, v := range D.Keys {
			s := fmt.Sprintf("%s", v)
			k = append(k, s)
		}
		ret = append(ret, strings.Join(k, sep))

	}
	for i, v := range D.Data {
		dline := make([]string, 1, l+1)
		dline[0] = fmt.Sprintf("%d", D.getithLabel(i))
		for _, w := range v {
			s := fmt.Sprintf("%5.4f", w)
			dline = append(dline, s)

		}
		ret = append(ret, strings.Join(dline, sep))
	}
	return strings.Join(ret, "\n")

}

func csverror(err error, linenu int, line string) error {
	return fmt.Errorf("Can't read line %d in libSVM-formatted file, Error: %v, line: %s", linenu, err, line)
}

// reads a libSVM-formatted file and returns a DataBunch. It's a pretty poor reader right now
// as it doesn't support sparse-libSVM files.
func DataBunchFromCSVFile(filename string, hasHeader, hasLabels bool, separator ...rune) (*DataBunch, error) {
	sep := ','
	if len(separator) != 0 {
		sep = separator[0]
	}
	f, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	return ParseCSVFromReader(f, hasHeader, hasLabels, sep)
}

func recordsToData(records []string, haslab bool, fields int) (int, []float64, error) {
	label := -1
	var err error
	if fields < 0 {
		fields = 1
	}
	start := 0
	data := make([]float64, 0, fields)
	if haslab {
		label, err = strconv.Atoi(records[0])
		if err != nil {
			return -1, nil, err
		}
		start++
	}
	for _, v := range records[start:] {
		f, err := strconv.ParseFloat(v, 64)
		if err != nil {
			return -1, nil, err
		}
		data = append(data, f)

	}

	return label, data, nil
}

func ParseCSVFromReader(r io.Reader, hasHeader, hasLabels bool, sep rune) (*DataBunch, error) {
	//	buf := bufio.NewReader(r)
	var err2 error
	var rec []string
	var headers []string
	var data [][]float64 = make([][]float64, 0, 1)
	var labels []int = make([]int, 0, 0)
	read := csv.NewReader(r)
	read.Comma = sep
	read.TrimLeadingSpace = true
	if hasHeader {
		recs, err := read.Read()
		if err != nil {
			return nil, err
		}

		ini := 0
		if hasLabels {
			ini = 1
		}
		headers = recs[ini:]

	}
	n := 0
	for {
		rec, err2 = read.Read()
		if err2 != nil {
			break
		}
		l, d, err3 := recordsToData(rec, hasLabels, n)
		if err3 != nil {

			return nil, errors.Join(fmt.Errorf("Posibly an issue with the header?"), err3)
		}
		if hasLabels {
			labels = append(labels, l)
		}
		n = len(d)
		data = append(data, d)

	}
	if !errors.Is(err2, io.EOF) {
		return nil, err2
	}

	return &DataBunch{Data: data, Labels: labels, Keys: headers}, nil
}
