package gb

import (
	"fmt"
	"math"

	"github.com/rmera/learn"
	"gonum.org/v1/gonum/floats"
)

type Tree struct {
	id                int //only trees read from files have id
	x                 [][]float64
	y                 []float64
	samples           []int
	value             float64
	nsamples          int //n
	features          int //c
	splitFeatureIndex int
	bestScoreSoFar    float64
	threshold         float64
	Left              *Tree
	Right             *Tree
	branches          int
}

type TreeOptions struct {
	MinChildWeight float64
	Indexes        []int
	MaxDepth       int
}

func DefaultTreeOptions() *TreeOptions {
	return &TreeOptions{MinChildWeight: 1, Indexes: nil, MaxDepth: 4}

}
func (T *TreeOptions) clone() *TreeOptions {
	ret := &TreeOptions{}
	ret.MinChildWeight = T.MinChildWeight
	ret.Indexes = T.Indexes //The idea of this method is precisely to keep the same options while changing the indexes, so
	//I could have assigned nil here. Still, the name "clone" suggests a full clone so here it is. Note that its the same
	//reference in both variables, as slices are pointers.
	return ret
}

func NewTree(X [][]float64, y []float64, options ...*TreeOptions) *Tree {
	var o *TreeOptions
	if len(options) > 0 && options[0] != nil {
		o = options[0]
	} else {
		o = DefaultTreeOptions()
	}
	if o.Indexes == nil {
		o.Indexes = make([]int, 0, len(X))
		for i := range X {
			o.Indexes = append(o.Indexes, i)
		}
	}

	ret := &Tree{}
	ret.samples = o.Indexes
	//	fmt.Println([]float64{1, 2, 3, 4, 5, 6, 7, 8}, []int{0, 3, 6}, learn.SampleSlice([]float64{1, 2, 3, 4, 5, 6, 7, 8}, []int{0, 3, 6}))
	sam := learn.SampleSlice(y, o.Indexes)
	ret.value = floats.Sum(sam) / float64(len(sam))
	ret.nsamples = len(o.Indexes)
	ret.features = len(X[0]) //no col subsampling
	ret.x = X
	ret.y = y
	ret.bestScoreSoFar = math.Inf(1)
	ret.branches = 1
	if o.MaxDepth > 0 {
		ret.maybeInsertChildNode(o)
	}
	return ret
}

func (T *Tree) maybeInsertChildNode(o *TreeOptions) {
	for i := 0; i < T.features; i++ {
		T.findBetterSplit(i, o)
	}
	if T.Leaf() {
		return
	}
	x := learn.SampleMatrix(T.x, o.Indexes, []int{T.splitFeatureIndex})
	indexleft := make([]int, 0, 3)
	indexright := make([]int, 0, 3)

	for i, v := range learn.TransposeFloats(x)[0] {
		if v <= T.threshold {
			indexleft = append(indexleft, o.Indexes[i])
		} else {
			indexright = append(indexright, o.Indexes[i])
		}
	}
	oleft := o.clone()
	oleft.MaxDepth--
	oright := oleft.clone()
	oleft.Indexes = indexleft
	oright.Indexes = indexright
	T.Left = NewTree(T.x, T.y, oleft)
	T.branches += T.Left.branches
	T.Right = NewTree(T.x, T.y, oright)
	T.branches += T.Right.branches
}

func (T *Tree) findBetterSplit(featureIndex int, o *TreeOptions) {
	x := learn.SampleMatrix(T.x, o.Indexes, []int{featureIndex})
	xt := learn.TransposeFloats(x) //x is a col vector
	sorted_indexes, sortx := learn.ArgSort(xt[0])
	ypart := learn.SampleSlice(T.y, o.Indexes)
	sorty := learn.SampleSlice(ypart, sorted_indexes)
	sumy := floats.Sum(sorty)
	sumyLeft, sumyRight := 0.0, sumy
	nleft := 0
	nright := T.nsamples
	sq := func(x float64) float64 { return x * x }
	for i := 0; i < T.nsamples-1; i++ {
		yi, xi, xinext := sorty[i], sortx[i], sortx[i+1]
		sumyLeft += yi
		sumyRight -= yi
		nright--
		nleft++
		if nleft < int(o.MinChildWeight) || xi == xinext {
			continue
		}
		if nright < int(o.MinChildWeight) {
			//	fmt.Println("minimum ny reached at i", i, nright, o.MinChildWeight) ///////////////////
			break
		}
		gain := -sq(sumyLeft)/float64(nleft) - sq(sumyRight)/float64(nright) + sq(sumy)/float64(T.nsamples)
		if gain < T.bestScoreSoFar {
			T.splitFeatureIndex = featureIndex
			T.bestScoreSoFar = gain
			T.threshold = (xi + xinext) / 2
		}

	}

}

func (T *Tree) Branches() int {
	return T.branches
}
func (T *Tree) Leaf() bool {
	return math.IsInf(T.bestScoreSoFar, 1)
}

func (T *Tree) Predict(data [][]float64, preds []float64) []float64 {
	if preds == nil {
		preds = make([]float64, len(data))
	}
	for i, v := range data {
		preds[i] = T.PredictSingle(v)
	}
	return preds
}

func (T *Tree) PredictSingle(row []float64) float64 {
	if T.Leaf() {
		return T.value
	}
	var child *Tree
	if row[T.splitFeatureIndex] <= T.threshold {
		child = T.Left
	} else {
		child = T.Right
	}
	return child.PredictSingle(row)
}

// If given the featurenames, returns the name of the split feature for the node. If not,
// returns the zero-based index for the split feature.
func (T *Tree) feature(featurenames []string) string {
	if featurenames == nil && len(featurenames) <= T.splitFeatureIndex {
		return fmt.Sprintf("%2d", T.splitFeatureIndex)
	}
	return featurenames[T.splitFeatureIndex]
}

// Will try to print the tree recursively. Taken from golearn (thanks!)
func (T *Tree) Print(spacing string, featurenames ...[]string) string {
	var featnames []string
	if len(featurenames) > 0 {
		featnames = featurenames[0]
	}
	if T.Leaf() {
		returnString := "  " + spacing + "PREDICT    "
		returnString += fmt.Sprintf("%.3f with %d Samples", T.value, T.samples) + "\n"
		return returnString

	}

	returnString := ""
	returnString += spacing + "Feature "
	returnString += T.feature(featnames)
	returnString += " < "
	returnString += fmt.Sprintf("%.3f", T.threshold)
	returnString += "\n"

	returnString += spacing + "---> True" + "\n"
	returnString += T.Left.Print(spacing+"  ", featurenames...)

	returnString += spacing + "---> False" + "\n"
	returnString += T.Right.Print(spacing+"  ", featurenames...)

	return returnString
}
