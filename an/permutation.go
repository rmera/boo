package an

import (
	"fmt"
	"math"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

func VariableImportance(xgb *boo.MultiClass, D *utils.DataBunch, feature *utils.IDOrKey) ([3]float64, error) {
	IniAcc := xgb.Accuracy(D)
	scdata, err := utils.PermuteFeatures(D, feature)
	if err != nil {
		return [3]float64{0, 0, 0}, fmt.Errorf("VariableImportance: Couldn't do permutation: %w", err)
	}
	ScAcc := xgb.Accuracy(scdata)
	return [3]float64{IniAcc - ScAcc, IniAcc, ScAcc}, nil
}

// A simple non-parametric function for p-value. The fraction of nulls with
// values at least as (default) or more extreme than score (if at least one moreextrem given and true).
func nonparam(score float64, nulls []float64, moreextreme ...bool) float64 {
	//	slices.Sort(nulls)
	var l, g int
	g = len(nulls)

	comp := func(score, v float64) bool {
		return score > v
	}
	if len(moreextreme) > 0 && moreextreme[0] {
		comp = func(score, v float64) bool {
			return score >= v
		}
	}
	for _, v := range nulls {
		if comp(score, v) {
			g--
			l++
		}
	}
	p := float64(g)
	return p / float64(len(nulls))
}

// This is an implementation of Altmann et al. Permutation Importance method.
// If you use this function, please cite:
// Altmann, André, Laura Toloşi, Oliver Sander, and Thomas Lengauer. "Permutation importance:
// a corrected feature importance measure." Bioinformatics 26, no. 10 (2010): 1340-1347.
// https://doi.org/10.1093/bioinformatics/btq13..

func PermutationImportance(training, test *utils.DataBunch, features *utils.IDOrKey, o *PermImportanceOptions) (float64, float64, error) {
	//I'm so not calling this function 'pimp'

	//only defined for >=1
	if o.Replicas <= 0 {
		o.Replicas = 1
	}
	if o.BooOpts == nil {
		return -1, -1, fmt.Errorf("PermutationImportance: no boo.Options given; supply the options used to train the model")
	}
	RefSample := 10
	if o.Replicas > RefSample {
		RefSample = o.Replicas
	}
	refscore := 0.0
	for i := 0; i < RefSample; i++ {

		xgb := boo.NewMultiClass(training, o.BooOpts)
		rs, err := o.Score(test, xgb, features)
		if err != nil {
			return -1, -1, fmt.Errorf("PermutationImportance: %w", err)
		}
		refscore += rs
	}
	refscore /= float64(RefSample)

	const epsilon float64 = 0.001
	if math.Abs(refscore) <= epsilon {
		return 0.0, -1.0, nil
	}

	nulls := make([]float64, 0, o.LabelPerms)
	for i := 0; i < o.LabelPerms; i++ {
		ns := 0.0
		ND := utils.PermuteLabels(training)
		for j := 0; j < o.Replicas; j++ {
			model := boo.NewMultiClass(ND, o.BooOpts)
			nns, err := o.Score(test, model, features)
			if err != nil {
				return -1, -1, err
			}
			ns += nns
		}
		nulls = append(nulls, ns/float64(o.Replicas))
	}
	return refscore, nonparam(refscore, nulls), nil
}

type PermImportanceOptions struct {
	LabelPerms int
	BooOpts    *boo.Options
	Replicas   int
	Score      func(*utils.DataBunch, *boo.MultiClass, *utils.IDOrKey) (float64, error) //I need that this can be nil
}

func DefaultPermImportanceOptions() *PermImportanceOptions {
	r := new(PermImportanceOptions)
	r.LabelPerms = 100
	r.Replicas = 1
	r.Score = func(test *utils.DataBunch, m *boo.MultiClass, feat *utils.IDOrKey) (float64, error) {
		FeatPerms := 100
		score := 0.0
		for i := 0; i < FeatPerms; i++ {
			res, err := VariableImportance(m, test, feat)
			if err != nil {
				return -1, fmt.Errorf("PermutationImportance: Score function failed: %w", err)
			}
			score += res[0]
		}
		return score / float64(FeatPerms), nil
	}
	return r
}
