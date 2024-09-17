package boo

// Based on the python code in https://randomrealizations.com/
// by Matt Bowers (https://github.com/mcb00)

import (
	"fmt"

	"github.com/rmera/boo/utils"
	"gonum.org/v1/gonum/mat"
)

// MultiClass is a multi-class gradient-boosted (xgboost or "regular")
// classification ensemble.
type MultiClass struct {
	b             [][]*Tree
	learningRate  float64
	classLabels   []int
	probTransform func(*mat.Dense, *mat.Dense) *mat.Dense
	tmp           []float64
	predtmp       []float64
	baseScore     float64
	xgb           bool
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

// Returns the number of "rounds" per class, in the given class,
// or, if no argument is given, in the first one (the rounds might not be all the same in all classes)
// ir the class index given is out of range, Rounds returns -1.
func (M *MultiClass) Rounds(class ...int) int {
	c := 0
	if len(class) > 0 && class[0] >= 0 {
		c = class[0]
	}
	//	println(len(M.b), c) /////////////////////////
	if M.Classes() < c || len(M.b) <= c {
		//println(len(M.b), c, M.Classes(), "v2") /////////////////////////

		return -1
	}
	return len(M.b[c])
}

// Returns the number of classes, i.e. the number of categories to which
// each data vector could belong.
func (M *MultiClass) Classes() int {
	return len(M.b)
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
			tmp[class] += tree.PredictSingle(instance) * M.learningRate
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
