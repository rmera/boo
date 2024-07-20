package gb

// Copyright (c) 2024 Raul Mera A.

import (
	"fmt"

	"github.com/rmera/learn"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type MultiClass struct {
	b             [][]*learn.Tree
	learningRate  float64
	maxDepth      int
	classLabels   []int
	probTransform func(*mat.Dense, *mat.Dense) *mat.Dense
	tmp           []float64
	predtmp       []float64
}

func (B *MultiClass) Accuracy(D *learn.DataBunch, classes ...int) float64 {
	right := 0
	instances := D.Data
	actualclasses := D.Labels
	if len(classes) > 0 && classes[0] > 0 && len(B.predtmp) < classes[0] {
		B.predtmp = make([]float64, classes[0])
	}

	for i, v := range instances {
		p := B.PredictSingleClass(v, B.predtmp)
		if B.classLabels[p] == actualclasses[i] {
			right++
		}
	}
	return 100.0 * (float64(right) / float64(len(instances)))

}

func (B *MultiClass) PredictSingleClass(instance []float64, predictions ...[]float64) int {
	preds := B.PredictSingle(instance, predictions...)
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

func (B *MultiClass) PredictSingle(instance []float64, predictions ...[]float64) []float64 {
	var preds []float64
	preds = make([]float64, len(B.b[0]))
	tmp := make([]float64, len(B.b[0]))
	for _, ensemble := range B.b {
		for class, tree := range ensemble {
			tmp[class] += tree.PredictSingle(instance) * B.learningRate
		}
	}
	O := mat.NewDense(1, len(tmp), tmp)
	D := mat.NewDense(1, len(preds), preds)
	D = B.probTransform(O, D)
	preds = D.RawMatrix().Data
	return preds
}

type Options struct {
	Rounds         int
	MaxDepth       int
	MinChildWeight int
	LearningRate   float64
	Loss           learn.LossFunc
}

func DefaultOptions() *Options {
	O := new(Options)
	O.Rounds = 10
	O.MaxDepth = 4
	O.LearningRate = 0.1
	O.MinChildWeight = 3
	O.Loss = &learn.MSELoss{}

	return O
}

func (O *Options) String() string {
	return fmt.Sprintf("%d r/%d md/%.3f lr", O.Rounds, O.MaxDepth, O.LearningRate)
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
	probs := learn.SoftMaxDense(rawPred, nil)
	ngrads := mat.NewDense(1, r, nil)
	hess := mat.NewDense(1, r, nil)
	tmpPreds := make([]float64, r) //not completely sure about this dimension
	for round := 0; round < O.Rounds; round++ {
		classes := make([]*learn.Tree, 0, 1)
		for k := 0; k < nlabels; k++ {
			//	fmt.Println("round, class", round, k)
			kthlabelvector := learn.DenseCol(ohelabels, k)
			kthprobs := learn.DenseCol(probs, k)
			ngrads = O.Loss.NegGradients(kthlabelvector, kthprobs, ngrads)
			hess = O.Loss.Hessian(kthprobs, hess)
			o := learn.DefaultGTreeOptions()
			o.MaxDepth = O.MaxDepth
			o.Y = ngrads.RawRowView(0)
			tree := learn.NewTree(D.Data, o)
			updateLeaves(tree, ngrads, hess)
			tmpPreds = tree.Predict(D.Data, tmpPreds)
			floats.Scale(O.LearningRate, tmpPreds)
			learn.AddToCol(rawPred, tmpPreds, k)
			probs = learn.SoftMaxDense(rawPred, probs)
			classes = append(classes, tree)
		}
		boosters = append(boosters, classes)
	}
	return &MultiClass{b: boosters, learningRate: O.LearningRate, maxDepth: O.MaxDepth, probTransform: learn.SoftMaxDense, classLabels: differentlabels}

}

func updateLeaves(tree *learn.Tree, gradient, hessian *mat.Dense) {
	fn := func(leaf *learn.Tree) {
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

func applyToLeafs(tree *learn.Tree, fn func(*learn.Tree)) {

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
