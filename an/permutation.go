package an

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

// Returns a copy of D with the Labels randomly permuted (shuffled).
// D.Labels must be filled (len>0). If FloatLabels is filled, its lenght must
// match that of D.Labels
func PermuteLabels(D *utils.DataBunch) *utils.DataBunch {
	D2 := D.Copy()
	perm := rand.Perm(len(D.Data))
	if len(D.Labels) == len(perm) {
		for i, v := range D.Labels {
			D2.Labels[perm[i]] = v
		}
	}
	if len(D.FloatLabels) == len(perm) {
		for i, v := range D.FloatLabels {
			D2.FloatLabels[perm[i]] = v
		}
	}
	return D2
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

func PermutationImportance(training, test *utils.DataBunch, features *IDOrKey, o *PermImportanceOptions) (float64, float64, error) {
	//I'm so not calling this function 'pimp'

	if o.BooOpts == nil {
		return -1, -1, fmt.Errorf("PermutationImportance: no boo.Options given; supply the options used to train the model")
	}
	RefSample := 10
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
		ND := PermuteLabels(training)
		model := boo.NewMultiClass(ND, o.BooOpts)
		ns, err := o.Score(test, model, features)
		if err != nil {
			return -1, -1, err
		}
		nulls = append(nulls, ns)
	}
	return refscore, nonparam(refscore, nulls), nil
}

type PermImportanceOptions struct {
	LabelPerms int
	BooOpts    *boo.Options
	Score      func(*utils.DataBunch, *boo.MultiClass, *IDOrKey) (float64, error) //I need that this can be nil
}

func DefaultPermImportanceOptions() *PermImportanceOptions {
	r := new(PermImportanceOptions)
	r.LabelPerms = 100
	r.Score = func(test *utils.DataBunch, m *boo.MultiClass, feat *IDOrKey) (float64, error) {
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
