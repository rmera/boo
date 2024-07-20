package learn

import (
	"fmt"
	"math"

	"github.com/rmera/chemlearn/utils"
	"gonum.org/v1/gonum/floats"
)

type Tree struct {
	id                int //only trees read from files have id
	grads             []float64
	hess              []float64
	x                 [][]float64
	y                 []float64
	Samples           []int
	bestScoreSoFar    float64
	Value             float64
	nsamples          int //n
	features          int //c
	splitFeatureIndex int
	threshold         float64
	Left              *Tree
	Right             *Tree
	xgb               bool
	branches          int
}

type TreeOptions struct {
	XGB             bool
	MinChildWeight  float64
	RegLambda       float64
	Gamma           float64
	ColSampleByNode float64
	Gradients       []float64
	Hessian         []float64
	Y               []float64
	MaxDepth        int
	Indexes         []int
}

//move these 2 to each package

func DefaultGTreeOptions() *TreeOptions {
	return &TreeOptions{MinChildWeight: 1, Indexes: nil, MaxDepth: 4, XGB: false}
}

func DefaultXTreeOptions() *TreeOptions {
	return &TreeOptions{MinChildWeight: 1, RegLambda: 1.0, Gamma: 1.0, ColSampleByNode: 1.0, Indexes: nil, MaxDepth: 4, XGB: true}

}

func (T *TreeOptions) clone() *TreeOptions {
	ret := &TreeOptions{}
	ret.XGB = T.XGB
	ret.Gradients = T.Gradients
	ret.Hessian = T.Hessian
	ret.Y = T.Y
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

func NewTree(X [][]float64, o *TreeOptions) *Tree {
	ret := &Tree{}
	if o.XGB {
		if o.Gradients == nil && o.Hessian == nil {
			panic("nil gradients/hessians in XGBoost tree")
		}
		ret.xgb = true
	}
	if o.Indexes == nil {
		o.Indexes = make([]int, 0, len(X))
		for i := range X {
			o.Indexes = append(o.Indexes, i)
		}
	}
	ret.Samples = o.Indexes
	ret.grads = o.Gradients
	ret.hess = o.Hessian
	ret.y = o.Y
	if ret.xgb {
		ret.Value = -1 * floats.Sum(utils.SampleSlice(ret.grads, o.Indexes)) / (floats.Sum(utils.SampleSlice(ret.hess, o.Indexes)) + o.RegLambda) //eq 5
		ret.bestScoreSoFar = 0.0
	} else {
		ret.bestScoreSoFar = math.Inf(1)
		sam := utils.SampleSlice(ret.y, o.Indexes)
		ret.Value = floats.Sum(sam) / float64(len(sam))

	}
	ret.nsamples = len(o.Indexes)
	ret.features = len(X[0]) //no col subsampling
	ret.x = X
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
	x := utils.SampleMatrix(T.x, o.Indexes, []int{T.splitFeatureIndex})
	indexleft := make([]int, 0, 3)
	indexright := make([]int, 0, 3)

	for i, v := range utils.TransposeFloats(x)[0] {
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
	//	fmt.Println("indices quiu", oleft.Indexes, oright.Indexes) /////////////////////
	T.Left = NewTree(T.x, oleft)
	T.branches += T.Left.branches
	T.Right = NewTree(T.x, oright)
	T.branches += T.Right.branches
}

func (T *Tree) findBetterSplit(featureIndex int, o *TreeOptions) {
	//	fmt.Println(T.x, o.Indexes, featureIndex) /////////////////////
	x := utils.SampleMatrix(T.x, o.Indexes, []int{featureIndex})
	xt := utils.TransposeFloats(x) //x is a col vector
	sorted_indexes, sortx := utils.ArgSort(xt[0])
	var xi, xinext, yi, gi, hi float64
	var g, h, sortg, sorth, ypart, sorty []float64
	var sumg, sumh, sumhRight, sumhLeft float64
	var sumgRight, sumgLeft, gain float64
	var sumy, sumyRight, sumyLeft float64
	var nleft, nright int = 0, T.nsamples
	var criterion func() bool

	if T.xgb {
		g = utils.SampleSlice(T.grads, o.Indexes)
		h = utils.SampleSlice(T.hess, o.Indexes)
		sortg = utils.SampleSlice(g, sorted_indexes)
		sorth = utils.SampleSlice(h, sorted_indexes)
		sumg, sumh = floats.Sum(g), floats.Sum(h)
		sumhRight, sumgRight = sumh, sumg
		sumhLeft, sumgLeft = 0.0, 0.0
	} else {
		ypart = utils.SampleSlice(T.y, o.Indexes)
		sorty = utils.SampleSlice(ypart, sorted_indexes)
		sumy = floats.Sum(sorty)
		sumyLeft, sumyRight = 0.0, sumy
	}
	sq := func(x float64) float64 { return x * x }
	for i := 0; i < T.nsamples-1; i++ {
		nright--
		nleft++
		if T.xgb {
			gi, hi, xi, xinext = sortg[i], sorth[i], sortx[i], sortx[i+1]

			sumgLeft += gi
			sumgRight -= gi
			sumhLeft += hi
			sumhRight -= hi
			//NOTE: this is not the actual meaning of the minchildweight in xgboost, but it
			//coincides with the current error function. I should probably change it to the
			//proper value.
			if nleft < int(o.MinChildWeight) || xi == xinext {
				continue
			}
			if nright < int(o.MinChildWeight) {
				break
			}
			gain = 0.5*((sq(sumgLeft)/(sumhLeft+o.RegLambda))+(sq(sumgRight)/(sumhRight+o.RegLambda))-(sq(sumg)/(sumh+o.RegLambda))) - (o.Gamma / 2) // Eq(7) in the xgboost paper
			//in eq 7 ,gamma is NOT divided by 2. Check!
			criterion = func() bool { return gain > T.bestScoreSoFar }

		} else {
			yi, xi, xinext = sorty[i], sortx[i], sortx[i+1]
			sumyLeft += yi
			sumyRight -= yi
			if nleft < int(o.MinChildWeight) || xi == xinext {
				continue
			}
			if nright < int(o.MinChildWeight) {
				//	fmt.Println("minimum ny reached at i", i, nright, o.MinChildWeight) ///////////////////
				break
			}
			gain = -sq(sumyLeft)/float64(nleft) - sq(sumyRight)/float64(nright) + sq(sumy)/float64(T.nsamples)
			criterion = func() bool { return gain < T.bestScoreSoFar }

		}
		if criterion() {
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
	if T.xgb {
		//	fmt.Println("return for xgb", T.bestScoreSoFar == 0) ////////////
		return T.bestScoreSoFar == 0
	} else {
		//	fmt.Println("return for no-xgb", math.IsInf(T.bestScoreSoFar, 1)) ////////////

		return math.IsInf(T.bestScoreSoFar, 1)
	}
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
		return T.Value
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
		returnString += fmt.Sprintf("%.3f with %d Samples", T.Value, T.Samples) + "\n"
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
