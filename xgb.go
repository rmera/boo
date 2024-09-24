package boo

// Copyright (c) 2024 Raul Mera A.

// Based on the python code in https://randomrealizations.com/
// by Matt Bowers (https://github.com/mcb00)

import (
	"fmt"
	"log"
	"math"
	"math/rand/v2"

	"github.com/rmera/boo/utils"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// Produces (and fits) a new multi-class classification boosted tree ensamble
// It will be of xgboost type if xgboost is true, regular gradient boosting othewise.
func NewMultiClass(D *utils.DataBunch, opts ...*Options) *MultiClass {
	var O *Options
	if len(opts) > 0 && opts[0] != nil {
		O = opts[0]
	} else {
		O = DefaultXOptions()
	}
	var ohelabels *mat.Dense
	var differentlabels []int
	actifunc := utils.SoftMaxDense
	if O.Regression {
		actifunc = utils.DoNothingDense
		ohelabels = D.LabelsRegression()
		differentlabels = []int{0}
	} else {
		ohelabels, differentlabels = D.OHELabels()
	}
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
	tin := make([]int, len(D.Data))
	tval := make([]float64, len(D.Data))
	probs := actifunc(rawPred, nil)
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
				tOpts.in = tin
				tOpts.val = tval
				tree = NewTree(D.Data, tOpts)
			} else {
				tOpts = DefaultGTreeOptions()
				tOpts.in = tin
				tOpts.val = tval
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
			probs = actifunc(rawPred, probs)
			var currloss float64
			if O.EarlyStop > 0 || O.Verbose {
				//    t:=mat.NewDense(1, len(tmpPreds), tmpPreds)
				kthprobs := utils.DenseCol(probs, k)
				currloss = O.Loss.Loss(kthlabelvector, kthprobs, tmploss)
			}
			classes = append(classes, tree)
			if O.Verbose {
				fmt.Printf("round: %d, class: %d train loss = %.3f\n", round, k, currloss)
			}
			if O.EarlyStop > 0 {
				epsilon := 1e-6
				if currloss <= epsilon {
					stopped[k] = true
					continue
				}
				if round == 0 {
					prevloss[k] = currloss
					continue
				}
				//	fmt.Println("losses", prevloss[k], currloss, roundsNoProgress[k], k) ///////////////
				if prevloss[k] <= currloss {
					roundsNoProgress[k]++
				} else {
					roundsNoProgress[k] = 0

				}
				if roundsNoProgress[k] >= O.EarlyStop {
					if O.Verbose {
						log.Println("Class", k, "stopped early at round", round)
					}
					stopped[k] = true
				}
				prevloss[k] = currloss
			}
		}
		boosters = append(boosters, classes)
	}
	return &MultiClass{b: boosters, learningRate: O.LearningRate, probTransform: actifunc, classLabels: differentlabels, baseScore: O.BaseScore, xgb: O.XGB, regression: O.Regression}

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
