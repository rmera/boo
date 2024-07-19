package xgb

import (
	"fmt"

	"github.com/rmera/learn"
	"gonum.org/v1/gonum/floats"
)

type Tree struct {
	id                int //only trees read from files have id
	grads             []float64
	hess              []float64
	x                 [][]float64
	y                 []float64
	samples           []int
	bestScoreSoFar    float64
	value             float64
	nsamples          int //n
	features          int //c
	splitFeatureIndex int
	threshold         float64
	Left              *Tree
	Right             *Tree
	branches          int
}

type TreeOptions struct {
	MinChildWeight  float64
	RegLambda       float64
	Gamma           float64
	ColSampleByNode float64
	MaxDepth        int
	Indexes         []int
}

func DefaultTreeOptions() *TreeOptions {
	return &TreeOptions{MinChildWeight: 1.0, RegLambda: 1.0, Gamma: 1.0, ColSampleByNode: 1.0, Indexes: nil, MaxDepth: 4}

}
func (T *TreeOptions) clone() *TreeOptions {
	ret := &TreeOptions{}
	ret.MinChildWeight = T.MinChildWeight
	ret.RegLambda = T.RegLambda
	ret.Gamma = T.Gamma
	ret.ColSampleByNode = T.ColSampleByNode
	ret.MaxDepth = T.MaxDepth
	ret.Indexes = T.Indexes //The idea of this method is precisely to keep the same options while changing the indexes, so
	//I could have assigned nil here. Still, the name "clone" suggests a full clone so here it is. Note that its the same
	//reference in both variables, as slices are pointers.
	return ret
}

func NewTree(X [][]float64, y, grads, hess []float64, options ...*TreeOptions) *Tree {
	var o *TreeOptions
	if len(options) > 0 && options[0] != nil {
		o = options[0]
	} else {
		o = DefaultTreeOptions()
	}

	if o.Indexes == nil {
		o.Indexes = make([]int, 0, len(grads))
		for i := range grads {
			o.Indexes = append(o.Indexes, i)
		}
	}
	ret := &Tree{}
	ret.samples = o.Indexes
	ret.grads = grads
	ret.hess = hess
	ret.value = -1 * floats.Sum(learn.SampleSlice(grads, o.Indexes)) / (floats.Sum(learn.SampleSlice(hess, o.Indexes)) + o.RegLambda) //eq 5

	ret.nsamples = len(o.Indexes)
	ret.features = len(X[0]) //no col subsampling
	ret.x = X
	ret.y = y
	ret.bestScoreSoFar = 0.0
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
	T.Left = NewTree(T.x, T.y, T.grads, T.hess, oleft)
	T.branches += T.Left.branches
	T.Right = NewTree(T.x, T.y, T.grads, T.hess, oright)
	T.branches += T.Right.branches
}

func (T *Tree) findBetterSplit(featureIndex int, o *TreeOptions) {
	x := learn.SampleMatrix(T.x, o.Indexes, []int{featureIndex})
	xt := learn.TransposeFloats(x) //x is a col vector
	sorted_indexes, sortx := learn.ArgSort(xt[0])
	g := learn.SampleSlice(T.grads, o.Indexes)
	h := learn.SampleSlice(T.hess, o.Indexes)
	sortg := learn.SampleSlice(g, sorted_indexes)
	sorth := learn.SampleSlice(h, sorted_indexes)
	sumg, sumh := floats.Sum(g), floats.Sum(h)
	sumhRight, sumgRight := sumh, sumg
	sumhLeft, sumgLeft := 0.0, 0.0
	sq := func(x float64) float64 { return x * x }
	for i := 0; i < T.nsamples-1; i++ {
		gi, hi, xi, xinext := sortg[i], sorth[i], sortx[i], sortx[i+1]
		sumgLeft += gi
		sumgRight -= gi
		sumhLeft += hi
		sumhRight -= hi
		if i+1 < int(o.MinChildWeight) || xi == xinext {

			continue
		}
		if T.nsamples-i < int(o.MinChildWeight) {
			break
		}
		gain := 0.5*((sq(sumgLeft)/(sumhLeft+o.RegLambda))+(sq(sumgRight)/(sumhRight+o.RegLambda))-(sq(sumg)/(sumh+o.RegLambda))) - (o.Gamma / 2) // Eq(7) in the xgboost paper
		//in eq 7 ,gamma is NOT divided by 2. Check!
		if gain > T.bestScoreSoFar {
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
	return T.bestScoreSoFar == 0
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
