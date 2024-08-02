package learn

// Copyright (c) 2024 Raul Mera A.

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/rmera/chemlearn/utils"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// MultiClass is a multi-class gradient-boosted (xgboost or "regular")
// classification ensamble.
type MultiClass struct {
	b             [][]*Tree
	utilsingRate  float64
	classLabels   []int
	probTransform func(*mat.Dense, *mat.Dense) *mat.Dense
	tmp           []float64
	predtmp       []float64
	baseScore     float64
	xgb           bool
}

// Contain options to create a multi-class classification ensamble.
type Options struct {
	XGB            bool
	Rounds         int
	MaxDepth       int
	EarlyTerm      int //roundw without increased fit before we stop trying.
	LearningRate   float64
	Lambda         float64
	MinChildWeight float64
	Gamma          float64
	SubSample      float64
	ColSubSample   float64
	BaseScore      float64
	MinSample      int //the minimum samples in each tree
	TreeMethod     string
	//	EarlyStopRounds      int //stop after n consecutive rounds of no improvement. Not implemented yet.
	Verbose bool
	Loss    utils.LossFunc
}

// Returns a pointer to an Options structure with the default values
// for an XGBoost multi-class classification ensamble.
func DefaultXOptions() *Options {
	O := new(Options)
	O.XGB = true
	O.Rounds = 20
	O.SubSample = 0.8
	O.ColSubSample = 0.8
	O.Lambda = 1.5
	O.MinChildWeight = 3
	O.Gamma = 0.2
	O.MaxDepth = 5
	O.LearningRate = 0.3
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.EarlyTerm = 10
	O.Loss = &utils.SQErrLoss{}
	O.Verbose = false //just for clarity
	O.MinSample = 5
	return O
}

func (o *Options) Equal(O *Options) bool {
	if O.XGB != o.XGB {
		return false
	}
	if O.Rounds != o.Rounds {
		return false
	}
	if O.SubSample != o.SubSample {
		return false
	}
	if O.ColSubSample != o.ColSubSample {
		return false
	}
	if O.Lambda != o.Lambda {
		return false
	}
	if O.MinChildWeight != o.MinChildWeight {
		return false
	}
	if O.Gamma != o.Gamma {

		return false
	}
	if O.MaxDepth != o.MaxDepth {
		return false
	}
	if O.LearningRate != o.LearningRate {
		return false
	}
	if O.BaseScore != o.BaseScore {
		return false
	}
	if O.TreeMethod != "exact" {
		return false
	}
	if O.Loss != o.Loss {
		return false
	}
	if O.Verbose != o.Verbose {
		return false
	}
	if O.MinSample != o.MinSample {
		return false
	}
	return true
}

func (o *Options) Clone() *Options {
	O := new(Options)
	O.XGB = o.XGB
	O.Rounds = o.Rounds
	O.SubSample = o.SubSample
	O.ColSubSample = o.ColSubSample
	O.Lambda = o.Lambda
	O.MinChildWeight = o.MinChildWeight
	O.Gamma = o.Gamma
	O.MaxDepth = o.MaxDepth
	O.LearningRate = o.LearningRate
	O.BaseScore = o.BaseScore
	O.TreeMethod = "exact"
	O.Loss = o.Loss
	O.Verbose = o.Verbose
	O.MinSample = o.MinSample
	return O

}

// Returns a pointer to an Options structure with the default
// options a for regular gradient boosting multi-class classification
// ensamble.
func DefaultGOptions() *Options {
	O := new(Options)
	O.XGB = false
	O.Rounds = 10
	O.MaxDepth = 4
	O.LearningRate = 0.1
	O.MinChildWeight = 3
	O.Loss = &utils.MSELoss{}

	return O
}

// Returns a string representation of the options
func (O *Options) String() string {
	if O.XGB {
		return fmt.Sprintf("xgboost %d r/%d md/%.3f lr/%.3f ss/%.3f bs/%.3f gam/%.3f lam/%.3f mcw/%.3f css", O.Rounds, O.MaxDepth, O.LearningRate, O.SubSample, O.BaseScore, O.Gamma, O.Lambda, O.MinChildWeight, O.ColSubSample)
	} else {
		return fmt.Sprintf("gboost %d r/%d md/%.3f lr/%.3f mcw", O.Rounds, O.MaxDepth, O.LearningRate, O.MinChildWeight)

	}

}

func DefaultOptions() *Options {
	return DefaultXOptions()
}

// Produces (and fits) a new multi-class classification boosted tree ensamble
// It will be of xgboost type if xgboost is true, regular gradient boosting othewise.
func NewMultiClass(D *utils.DataBunch, opts ...*Options) *MultiClass {
	var O *Options
	if len(opts) > 0 && opts[0] != nil {
		O = opts[0]
	} else {
		O = DefaultXOptions()
	}
	ohelabels, differentlabels := D.OHELabels()
	nlabels := len(differentlabels)
	boosters := make([][]*Tree, 0, nlabels)
	r, c := ohelabels.Dims()
	rawPred := mat.NewDense(r, c, nil)
	utils.ToOnes(rawPred)
	rawPred.Scale(O.BaseScore, rawPred)
	//just to be safe
	if O.MinChildWeight < 1 {
		O.MinChildWeight = 1
	}
	probs := utils.SoftMaxDense(rawPred, nil)
	grads := mat.NewDense(1, r, nil)
	hess := mat.NewDense(1, r, nil)
	tmpPreds := make([]float64, r)
	tmploss := mat.NewDense(1, r, make([]float64, r))
	stopped := make([]bool, len(differentlabels))
	roundsNoProgress := make([]int, len(differentlabels))
	prevloss := make([]float64, len(differentlabels))

	for round := 0; round < O.Rounds; round++ {
		var sampleIndexes, sampleCols []int
		if O.SubSample < 1 && O.XGB {
			sampleIndexes = SubSample(len(D.Data), O.SubSample)
		}
		if O.ColSubSample < 1 && O.XGB {
			sampleCols = SubSample(len(D.Data[0]), O.ColSubSample)
		}
		if len(sampleIndexes) < O.MinSample {
			continue
		}
		classes := make([]*Tree, 0, 1)
		for k := 0; k < nlabels; k++ {
			if stopped[k] {
				continue
			}
			var tOpts *TreeOptions
			var tree *Tree
			kthlabelvector := utils.DenseCol(ohelabels, k)
			kthprobs := utils.DenseCol(probs, k)
			hess = O.Loss.Hessian(kthprobs, nil) //keep an eye on this.
			if O.XGB {
				tOpts = DefaultXTreeOptions()
				tOpts.MinChildWeight = O.MinChildWeight
				tOpts.MaxDepth = O.MaxDepth
				grads = O.Loss.Gradients(kthlabelvector, kthprobs, grads)
				tOpts.Lambda = O.Lambda
				tOpts.Gamma = O.Gamma
				tOpts.Indexes = sampleIndexes
				tOpts.AllowedColumns = sampleCols
				tOpts.Gradients = grads.RawRowView(0)
				tOpts.Hessian = hess.RawRowView(0)
				tOpts.Y = kthlabelvector.RawRowView(0)
				tree = NewTree(D.Data, tOpts)
			} else {
				tOpts = DefaultGTreeOptions()
				tOpts.MinChildWeight = O.MinChildWeight
				tOpts.MaxDepth = O.MaxDepth
				grads = O.Loss.NegGradients(kthlabelvector, kthprobs, grads)
				tOpts.Y = grads.RawRowView(0)
				tOpts.Gradients = nil
				tOpts.Hessian = nil
				tree = NewTree(D.Data, tOpts)
				updateLeaves(tree, grads, hess)
			}
			tmpPreds = tree.Predict(D.Data, tmpPreds)
			floats.Scale(O.LearningRate, tmpPreds)
			utils.AddToCol(rawPred, tmpPreds, k)
			probs = utils.SoftMaxDense(rawPred, probs)
			var currloss float64
			if O.EarlyTerm > 0 || O.Verbose {
				//    t:=mat.NewDense(1, len(tmpPreds), tmpPreds)
				kthprobs := utils.DenseCol(probs, k)
				currloss = O.Loss.Loss(kthlabelvector, kthprobs, tmploss)
			}
			classes = append(classes, tree)
			if O.Verbose {
				fmt.Printf("round: %d, class: %d train loss = %.3f\n", round, k, currloss)
			}
			if O.EarlyTerm > 0 {
				epsilon := 1e-6
				if currloss >= epsilon {
					stopped[k] = true
					continue
				}
				if round == 0 {
					prevloss[k] = currloss
					continue
				}
				if prevloss[k] <= currloss {
					roundsNoProgress[k]++
				}
				if roundsNoProgress[k] >= O.EarlyTerm {
					stopped[k] = true
				}
				prevloss[k] = currloss
			}
		}
		boosters = append(boosters, classes)
	}
	return &MultiClass{b: boosters, utilsingRate: O.LearningRate, probTransform: utils.SoftMaxDense, classLabels: differentlabels, baseScore: O.BaseScore, xgb: O.XGB}

}

// Obtains the Log of the odds for a nxm matrix
// where each element i,j is the probability of the
// samble i to belong to class j.
func LogOddsFromProbs(m *mat.Dense) *mat.Dense {
	r, c := m.Dims()
	ret := mat.NewDense(r, c, nil)
	for i := 0; i < r; i++ {
		row := m.RawRowView(i)
		s := floats.Sum(row)
		for j, v := range row {
			ret.Set(i, j, math.Log(v*(s-v)))
		}
	}
	return ret
}

// given an nxm matrix p, where n is the number of samples
// and n is the number of classes, and each element i,j is
// the probability of sample i of being in class j, returns
// a nx1 column matrix where each element corresponds to the
// most likely class for sample i (i.e., for each row, the
// column in the original matrix with the largest value.
func ClassesFromProbs(p *mat.Dense) *mat.Dense {
	r, _ := p.Dims()
	classes := mat.NewDense(r, 1, nil)
	for i := 0; i < r; i++ {
		row := p.RawRowView(i)
		maxi := row[0]
		maxin := 0
		for i, v := range row {
			if v > maxi {
				maxi = v
				maxin = i
			}
		}
		classes.Set(i, 0, float64(maxin))
	}
	return classes
}

// returns a slice with the indexes of a slice with total elements
// totaldata that are selected for sambling with a subsamble
// probability.
func SubSample(totdata int, subsample float64) []int {
	ret := make([]int, 0, int(float64(totdata)*subsample)+1)
	for i := 0; i < totdata; i++ {
		if subsample >= rand.Float64() {
			ret = append(ret, i)
		}
	}
	return ret
}

// Returns the percentage of accuracy of the model on the data (which needs to contain
// labels). You can give it the number of classes present, which helps with memory.
func (M *MultiClass) Accuracy(D *utils.DataBunch, classes ...int) float64 {
	right := 0
	instances := D.Data
	actualclasses := D.Labels
	if len(classes) > 0 && classes[0] > 0 && len(M.predtmp) < classes[0] {
		M.predtmp = make([]float64, classes[0])
	}
	for i, v := range instances {
		p := M.PredictSingleClass(v, M.predtmp)
		if M.classLabels[p] == actualclasses[i] {
			right++
		}
	}
	return 100.0 * (float64(right) / float64(len(instances)))

}

// Predicts the class to which a single sample belongs. You can give a slice of floats
// to use as temporal storage for the probabilities that are used to assign the class
func (M *MultiClass) PredictSingleClass(instance []float64, predictions ...[]float64) int {
	preds := M.PredictSingle(instance, predictions...)
	prediction := 0
	maxval := preds[0]
	for i, v := range preds {
		if v > maxval {
			prediction = i
			maxval = v
		}
	}
	return prediction
}

// Returns a slice with the probability of the sample belonging to each class. You can supply
// a slice to be filled with the predictions in order to avoid allocation.
func (M *MultiClass) PredictSingle(instance []float64, predictions ...[]float64) []float64 {
	var preds []float64
	preds = make([]float64, len(M.b[0]))
	tmp := make([]float64, len(M.b[0]))
	for i := range tmp {
		tmp[i] = M.baseScore
	}
	for _, ensemble := range M.b {
		for class, tree := range ensemble {
			tmp[class] += tree.PredictSingle(instance) * M.utilsingRate
		}
	}
	O := mat.NewDense(1, len(tmp), tmp)
	D := mat.NewDense(1, len(preds), preds)
	D = M.probTransform(O, D)
	preds = D.RawMatrix().Data
	return preds //SHOULD contain the numbers now.
}

// Returns the features ranked by their "importance" to the classification.
func (M *MultiClass) FeatureImportance() (*Feats, error) {
	ret := NewFeats(M.xgb)
	for round, ensemble := range M.b {
		for class, tree := range ensemble {
			_, err := tree.FeatureImportance(M.xgb, ret)
			if err != nil {
				return nil, fmt.Errorf("Error with features of tree for class %d, boosting round %d", class, round)
			}
		}
	}
	return ret, nil
}

func updateLeaves(tree *Tree, gradient, hessian *mat.Dense) {
	fn := func(leaf *Tree) {
		if leaf.samples == nil {
			panic("Samples in one leaf are nil!")
		}
		var sumhess, sumgrad float64
		for _, w := range leaf.samples {
			sumhess += hessian.At(0, w)
			sumgrad += gradient.At(0, w)
		}
		nval := sumgrad / sumhess
		leaf.value = nval
	}
	applyToLeafs(tree, fn)
}

func applyToLeafs(tree *Tree, fn func(*Tree)) {

	if tree.left != nil {
		applyToLeafs(tree.left, fn)
	}

	if tree.right != nil {
		applyToLeafs(tree.right, fn)
	}
	if tree.Leaf() {
		fn(tree)
		return
	}
}
