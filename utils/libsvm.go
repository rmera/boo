package utils

import (
	"bufio"
	"fmt"
	"io"
	"os"
	"slices"
	"strconv"
	"strings"
)

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
