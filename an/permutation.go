package an

import (
	"math/rand/v2"
	"slices"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

// Returns a copy of D with the Labels randomly permuted (shuffled).
func PermuteLabels(D *utils.DataBunch) *utils.DataBunch {
	D2 := D.Copy()
	nsamples := len(D2.Labels)
	used := make([]int, 0, nsamples)

	NewIndex := func() int {
		for {
			a := rand.IntN(nsamples)
			if slices.Contains(used, a) {
				continue
			}
			used = append(used, a)
			return a
		}
	}
	for _, v := range D.Labels {
		newindex := NewIndex()
		D2.Labels[newindex] = v
	}
	return D2
}

// A simple non-parametric function for p-value. The fraction of nulls with
// values more extreme than score.
func nonparam(score float64, nulls []float64, twotails ...bool) float64 {
	slices.Sort(nulls)
	var l, g int
	g = len(nulls)
	for _, v := range nulls {
		if score >= v {
			g--
			l++
		} else {
			break
		}
	}
	p := float64(g)
	if len(twotails) > 0 && twotails[0] {
		p = 2 * float64(min(g, l))
	}
	return p / float64(len(nulls))
}

// This is an implementation of Altmann et al. Permutation Importance method.
// If you use this function, please cite:
// Altmann, André, Laura Toloşi, Oliver Sander, and Thomas Lengauer. "Permutation importance:
// a corrected feature importance measure." Bioinformatics 26, no. 10 (2010): 1340-1347.
// https://doi.org/10.1093/bioinformatics/btq13
func PermutationImportance(xgb *boo.MultiClass, D *utils.DataBunch, features *IDOrKey, opts ...*PermImportanceOptions) (float64, error) {
	//I'm so not calling this 'pimp'
	//	nperms := 1000
	//	if len(npermut) > 0 && npermut[0] > 0 {
	//		nperms = npermut[0]
	//	}
	var o *PermImportanceOptions
	if len(opts) > 0 {
		o = opts[0]
	} else {
		o = DefaultPermImportanceOptions()
	}

	if _, ok := o.Score(); !ok {
		t, err := VariableImportance(xgb, D, features)
		if err != nil {
			return -1, err
		}
		o.Score(t[0])
	}

	nulls := make([]float64, 0, o.LabelPerms)
	for i := 0; i < o.LabelPerms; i++ {
		ND := PermuteLabels(D)
		ns, err := VariableImportance(xgb, ND, features)
		if err != nil {
			return -1, err
		}
		nulls = append(nulls, ns[0])
	}

	//NOTE: Add parametric versions (the reference has normal, log-normal and gamma, in addition to the non-parametric method)
	sc, _ := o.Score()
	return nonparam(sc, nulls, o.TwoTails), nil
}

type PermImportanceOptions struct {
	LabelPerms int
	//	FeatPerms  int
	score     float64 //I need that this can be nil
	havescore bool
	TwoTails  bool
}

// returns the score plus true if a score has been set and false if not.
// if  given a number it will return the existing information and then set the score
// to the value given.
func (P *PermImportanceOptions) Score(f ...float64) (float64, bool) {
	if len(f) != 0 {
		tr := P.score
		trb := P.havescore
		P.score = f[0]
		P.havescore = true
		return tr, trb
	}
	return P.score, P.havescore

}

func DefaultPermImportanceOptions() *PermImportanceOptions {
	r := new(PermImportanceOptions)
	r.LabelPerms = 10000
	//	r.FeatPerms = 100
	r.TwoTails = false
	return r

}
