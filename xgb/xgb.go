package xgb

// Copyright (c) 2024 Raul Mera A.

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/rmera/learn"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type MultiClass struct {
	b             [][]*learn.Tree
	learningRate  float64
	classLabels   []int
	probTransform func(*mat.Dense, *mat.Dense) *mat.Dense
	tmp           []float64
	predtmp       []float64
	baseScore     float64
}

type Options struct {
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
	Loss           learn.LossFunc
}

func DefaultOptions() *Options {
	O := new(Options)
	O.Rounds = 20
	O.SubSample = 0.8
	O.RegLambda = 1.5
	O.MinChildWeight = 3
	O.MaxDepth = 5
	O.LearningRate = 0.3
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.Loss = &learn.SQErrLoss{}
	O.Verbose = false //just for clarity
	O.MinSample = 5
	return O
}

func (O *Options) String() string {
	return fmt.Sprintf("%d r/%d md/%.3f lr/%.3f ss/%.3f bs/%.3f gam/%.3f lam/%.3f mcw/", O.Rounds, O.MaxDepth, O.LearningRate, O.SubSample, O.BaseScore, O.Gamma, O.RegLambda, O.MinChildWeight)
}

func NewMultiClass(D *learn.DataBunch, opts ...*Options) *MultiClass {
	var O *Options
	if len(opts) > 0 && opts[0] != nil {
		O = opts[0]
	} else {
		O = DefaultOptions()
	}
	ohelabels, differentlabels := D.OHELabels()
	nlabels := len(differentlabels)
	boosters := make([][]*learn.Tree, 0, nlabels)
	r, c := ohelabels.Dims()
	rawPred := mat.NewDense(r, c, nil)
	learn.ToOnes(rawPred)
	rawPred.Scale(O.BaseScore, rawPred)
	probs := learn.SoftMaxDense(rawPred, nil)
	grads := mat.NewDense(1, r, nil)
	hess := mat.NewDense(1, r, nil)
	tmpPreds := make([]float64, r)
	for round := 0; round < O.Rounds; round++ {
		var sampleIndexes []int
		if O.SubSample < 1 {
			sampleIndexes = SubSample(len(D.Data), O.SubSample)
		}
		if len(sampleIndexes) < O.MinSample {
			continue
		}
		classes := make([]*learn.Tree, 0, 1)
		for k := 0; k < nlabels; k++ {
			kthlabelvector := learn.DenseCol(ohelabels, k)
			kthrawpred := learn.DenseCol(rawPred, k)
			grads = O.Loss.Gradients(kthlabelvector, kthrawpred, nil) //here I could/should reuse the grads slice
			hess = O.Loss.Hessian(kthrawpred, nil)                    //same here, I can pass them instead of nil
			tOpts := learn.DefaultXTreeOptions()
			tOpts.MinChildWeight = O.MinChildWeight
			tOpts.RegLambda = O.RegLambda
			tOpts.Gamma = O.Gamma
			tOpts.MaxDepth = O.MaxDepth
			tOpts.Indexes = sampleIndexes
			tOpts.Gradients = grads.RawRowView(0)
			tOpts.Hessian = hess.RawRowView(0)
			tOpts.Y = kthlabelvector.RawRowView(0)
			tree := learn.NewTree(D.Data, tOpts)
			tmpPreds = tree.Predict(D.Data, tmpPreds)
			floats.Scale(O.LearningRate, tmpPreds)
			learn.AddToCol(rawPred, tmpPreds, k)
			probs = learn.SoftMaxDense(rawPred, probs)
			if O.Verbose {
				fmt.Printf("round: %d, class: %d train loss = %.3f", round, k, O.Loss.Loss(kthlabelvector, mat.NewDense(0, len(tmpPreds), tmpPreds), nil))
			}
			classes = append(classes, tree)
		}
		boosters = append(boosters, classes)
	}
	return &MultiClass{b: boosters, learningRate: O.LearningRate, probTransform: learn.SoftMaxDense, classLabels: differentlabels, baseScore: O.BaseScore}

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

func (M *MultiClass) Accuracy(D *learn.DataBunch, classes ...int) float64 {
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
			tmp[class] += tree.PredictSingle(instance) * M.learningRate
		}
	}
	O := mat.NewDense(1, len(tmp), tmp)
	D := mat.NewDense(1, len(preds), preds)
	D = M.probTransform(O, D)
	preds = D.RawMatrix().Data
	return preds //SHOULD contain the numbers now.
}
