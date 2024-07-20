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

type MultiClass struct {
	b             [][]*Tree
	utilsingRate  float64
	classLabels   []int
	probTransform func(*mat.Dense, *mat.Dense) *mat.Dense
	tmp           []float64
	predtmp       []float64
	baseScore     float64
}

type Options struct {
	XGB            bool
	Rounds         int
	MaxDepth       int
	LearningRate   float64
	RegLambda      float64
	MinChildWeight float64
	Gamma          float64
	SubSample      float64
	BaseScore      float64
	MinSample      int //the minimum samples in each tree
	TreeMethod     string
	Verbose        bool
	Loss           utils.LossFunc
}

func DefaultXOptions() *Options {
	O := new(Options)
	O.XGB = true
	O.Rounds = 20
	O.SubSample = 0.8
	O.RegLambda = 1.5
	O.MinChildWeight = 3
	O.MaxDepth = 5
	O.LearningRate = 0.3
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.Loss = &utils.SQErrLoss{}
	O.Verbose = false //just for clarity
	O.MinSample = 5
	return O
}

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

func (O *Options) String() string {
	if O.XGB {
		return fmt.Sprintf("xgboost %d r/%d md/%.3f lr/%.3f ss/%.3f bs/%.3f gam/%.3f lam/%.3f mcw/", O.Rounds, O.MaxDepth, O.LearningRate, O.SubSample, O.BaseScore, O.Gamma, O.RegLambda, O.MinChildWeight)
	} else {
		return fmt.Sprintf("gboost %d r/%d md/%.3f lr", O.Rounds, O.MaxDepth, O.LearningRate)

	}

}

func NewMultiClass(D *utils.DataBunch, xgboost bool, opts ...*Options) *MultiClass {
	var O *Options
	if len(opts) > 0 && opts[0] != nil {
		O = opts[0]
	} else {
		if xgboost {
			O = DefaultXOptions()
		} else {
			O = DefaultGOptions()
		}
	}
	O.XGB = xgboost
	ohelabels, differentlabels := D.OHELabels()
	nlabels := len(differentlabels)
	boosters := make([][]*Tree, 0, nlabels)
	r, c := ohelabels.Dims()
	rawPred := mat.NewDense(r, c, nil)
	utils.ToOnes(rawPred)
	rawPred.Scale(O.BaseScore, rawPred)
	probs := utils.SoftMaxDense(rawPred, nil)
	grads := mat.NewDense(1, r, nil)
	hess := mat.NewDense(1, r, nil)
	tmpPreds := make([]float64, r)
	for round := 0; round < O.Rounds; round++ {
		var sampleIndexes []int
		if O.SubSample < 1 && xgboost {
			sampleIndexes = SubSample(len(D.Data), O.SubSample)
		}
		if len(sampleIndexes) < O.MinSample {
			continue
		}
		classes := make([]*Tree, 0, 1)
		for k := 0; k < nlabels; k++ {
			var tOpts *TreeOptions
			var tree *Tree
			kthlabelvector := utils.DenseCol(ohelabels, k)
			kthrawpred := utils.DenseCol(rawPred, k)
			hess = O.Loss.Hessian(kthrawpred, nil) //same here, I can pass them instead of nil
			kthprobs := utils.DenseCol(probs, k)
			if O.XGB {
				tOpts = DefaultXTreeOptions()
				tOpts.MinChildWeight = O.MinChildWeight
				tOpts.MaxDepth = O.MaxDepth
				grads = O.Loss.Gradients(kthlabelvector, kthprobs, grads) //here I could/should reuse the grads slice
				tOpts.RegLambda = O.RegLambda
				tOpts.Gamma = O.Gamma
				tOpts.Indexes = sampleIndexes
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
			if O.Verbose {
				fmt.Printf("round: %d, class: %d train loss = %.3f", round, k, O.Loss.Loss(kthlabelvector, mat.NewDense(0, len(tmpPreds), tmpPreds), nil))
			}
			classes = append(classes, tree)
		}
		boosters = append(boosters, classes)
	}
	return &MultiClass{b: boosters, utilsingRate: O.LearningRate, probTransform: utils.SoftMaxDense, classLabels: differentlabels, baseScore: O.BaseScore}

}

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

func SubSample(totdata int, subsample float64) []int {
	ret := make([]int, 0, int(float64(totdata)*subsample)+1)
	for i := 0; i < totdata; i++ {
		if subsample >= rand.Float64() {
			ret = append(ret, i)
		}
	}
	return ret
}

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

func updateLeaves(tree *Tree, gradient, hessian *mat.Dense) {
	fn := func(leaf *Tree) {
		if leaf.Samples == nil {
			panic("Samples in one leaf are nil!")
		}
		var sumhess, sumgrad float64
		for _, w := range leaf.Samples {
			sumhess += hessian.At(0, w)
			sumgrad += gradient.At(0, w)
		}
		nval := sumgrad / sumhess
		leaf.Value = nval
	}
	applyToLeafs(tree, fn)
}

func applyToLeafs(tree *Tree, fn func(*Tree)) {

	if tree.Left != nil {
		applyToLeafs(tree.Left, fn)
	}

	if tree.Right != nil {
		applyToLeafs(tree.Right, fn)
	}
	if tree.Leaf() {
		fn(tree)
		return
	}
}
