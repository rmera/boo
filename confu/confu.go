package confu

import (
	"bufio"
	"errors"
	"fmt"
	"io"
	"os"
	"slices"
	"sort"
	"strconv"
	"strings"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

// Builds a basic dictionary where each label is simply
// given its numerical value as a name. Can be used if you don't
// have string names for your labels.
func MakeSimpleNameMap(labels []int) map[int]string {
	ret := make(map[int]string)
	for _, v := range labels {
		ret[v] = fmt.Sprintf("%d", v)
	}
	return ret
}

// Reads a map for the names of each label. The names file must have one entry per line.
// In that entry, there must be 2 fields separated by separator (a space by default)
// the first one is the number of the label, the second one, its name.
func ReadNameMap(filename string, separator ...string) (map[int]string, error) {
	sep := " "
	if len(separator) > 0 && separator[0] != "" {
		sep = separator[0]
	}
	ffin, err := os.Open(filename)
	if err != nil {
		return nil, err
	}
	defer ffin.Close()
	fin := bufio.NewReader(ffin)
	var rerr error
	ret := make(map[int]string)
	for s, rerr := fin.ReadString('\n'); rerr == nil; s, rerr = fin.ReadString('\n') {
		l := strings.Split(s, sep)
		num, err := strconv.Atoi(l[0])
		if err != nil {
			return nil, err
		}
		ret[num] = l[1]
	}
	if errors.Is(rerr, io.EOF) {
		rerr = nil //EOF is not an actual error
	}
	return ret, rerr

}

type Confusions struct {
	Actual    []int
	Predicted []int
	Labels    []int
	matrix    [][]int
	tmpvals   []int
	tmpindx   []int
	m         map[int]int //label to the index of the label in Confusions.Labels

}

func MCConfusions(M *boo.MultiClass, D *utils.DataBunch) *Confusions {
	instances := D.Data
	ret := &Confusions{Actual: D.Labels}
	_, ret.Labels = D.OHELabels()
	ret.Predicted = make([]int, 0, len(ret.Actual))
	for _, v := range instances {
		ret.Predicted = append(ret.Predicted, ret.Labels[M.PredictSingleClass(v)])
	}

	return ret
}

func (C *Confusions) Matrix() [][]int {
	if C.matrix != nil {
		return C.matrix
	}
	if C.m == nil {
		C.m = make(map[int]int)
		for i, v := range C.Labels {
			C.m[v] = i
		}

	}
	//	fmt.Println("MAPA", C.m, C.Predicted, C.Actual) ////////////////////////////////////////
	C.matrix = make([][]int, 0, len(C.Labels))
	for _, v := range C.Labels {
		lab := make([]int, 0, 10)
		//we collect the indexes of label v in the actual labels.
		//NOTE: I could make this more efficient
		for j, w := range C.Actual {
			if w == v {
				lab = append(lab, j)
			}
		}
		confRow := make([]int, len(C.Labels))
		for _, w := range lab {
			//			println("element", C.m[C.Predicted[w]], C.Predicted[w], w) ////////
			confRow[C.m[C.Predicted[w]]]++
		}
		C.matrix = append(C.matrix, confRow)
	}
	return C.matrix
}

func (C *Confusions) PrintTopN(N int, m map[int]string) string {
	r, rf := C.TopNPerLabel(N)
	ret := make([]string, 1, len(m)+1)
	ret[0] = "Confusions:"
	for i, v := range r {
		str := make([]string, 1, N)
		str[0] = fmt.Sprintf("Label %d", C.Labels[i])
		n := N
		if len(v) < N {
			n = len(v)
		}
		for j := 0; j < n; j++ {
			str = append(str, fmt.Sprintf("%d:%4.1f", C.Labels[v[j]], rf[i][j]*100))
		}
		ret = append(ret, strings.Join(str, ","))
	}
	return strings.Join(ret, "\n")

}

// For each label, returns the
func (C *Confusions) TopNPerLabel(N int) ([][]int, [][]float64) {
	ret := make([][]int, 0, len(C.Labels))
	retf := make([][]float64, 0, len(C.Labels))

	for _, v := range C.Labels {
		fmt.Println("label", v) //////////////////
		r, rf := C.TopNForLabelM(N, v, true)
		fmt.Println("weaita", r, rf) ///////////////////
		ret = append(ret, r)
		retf = append(retf, rf)
	}
	return ret, retf
}

// Returns the a []int with the M labels most commonly predicted by the model
// when the actual label is M. If M is not a valid label in the dataset,
// returns nil. It also returns the fraction of instances where label M is predicted to be
// each of the labels.
func (C *Confusions) TopNForLabelM(N, M int, docopy ...bool) ([]int, []float64) {
	C.Matrix()
	index := slices.Index(C.Labels, M)
	if index < 0 {
		return nil, nil
	}
	fmt.Println(C.matrix, "beg", "label and index", M, index, C.Labels) /////////
	om := C.matrix[index]

	if len(C.tmpvals) != len(om) {
		C.tmpvals = make([]int, len(om))
	}
	if len(C.tmpindx) != len(om) {
		C.tmpindx = make([]int, len(om))
	}

	sorti, sortvals := memIntSort(om, C.tmpindx, C.tmpvals)
	fmt.Println(om, "sorteds", sorti, sortvals) ///////////////////////////////////////////////////
	//now transform from index to actual labels
	//	for i:=range(sorti){
	//		sorti[i]=C.Labels[i]
	//	}
	sum := 0
	for _, v := range sortvals {
		sum += v
	}
	slices.Reverse(sorti)
	slices.Reverse(sortvals)
	if len(docopy) > 0 && docopy[0] {
		println("copiacu!") //////////////////////
		cpsorti := make([]int, len(sorti))
		cpsortvals := make([]int, len(sortvals))
		copy(cpsortvals, sortvals)
		copy(cpsorti, sorti)
		fmt.Println("copias", sorti, cpsorti, sortvals, cpsortvals) ///////////
		sorti = cpsorti
		sortvals = cpsortvals
	}
	fracs := func(N int, sortvals []int) []float64 {
		f := make([]float64, 0, N)
		for _, v := range sortvals[:N] {
			println("v,sum", v, sum) ///////////////////
			num := 0.0
			if v != 0 {
				num = float64(v) / float64(sum)
			}
			f = append(f, num)
		}
		return f
	}
	if N >= len(sorti) {
		return sorti, fracs(len(sortvals), sortvals)
	}
	return sorti[:N], fracs(N, sortvals)
}

type idSorter struct {
	i []int
	a []int
}

func (s *idSorter) Len() int           { return len(s.i) }
func (s *idSorter) Less(i, j int) bool { return s.a[i] < s.a[j] }
func (s *idSorter) Swap(i, j int) {
	s.a[i], s.a[j] = s.a[j], s.a[i]
	s.i[i], s.i[j] = s.i[j], s.i[i]
}

// Returns the indexes that would sort the given slice
// and also the sorted slice. It doesn't touch the original slice!
func memIntSort(a []int, in []int, val []int) ([]int, []int) {
	for i, v := range a {
		in[i] = i  // = append(in, i)
		val[i] = v // = append(val, v)
	}
	r := &idSorter{i: in, a: val}
	sort.Sort(r)
	return r.i, r.a
}
