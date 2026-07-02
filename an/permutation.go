package an

import (
	"math/rand/v2"
	"slices"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

// Returns D, not a copy, with the Labels permutefeatures in the given index set scrambled. All features in the given
// set are scrambled randomly but identically. i.e. using the same random permutation for both each time.
// it also takes 2 slices to be used as scratch (it allocates for them if they are nil) and returns 2 to be
// used in future calls. The second one contains the labels in the original order
func PermuteLabels(D *utils.DataBunch, used, tmp []int) (*utils.DataBunch, []int, []int) {
	samples := len(D.Labels)
	if used == nil {
		used = make([]int, 0, samples)
	} else {
		used = used[:0]

	}

	if len(tmp) != len(D.Labels) {
		tmp = make([]int, samples)
	}

	NewIndex := func() int {
		for {
			a := rand.IntN(samples)
			if slices.Contains(used, a) {
				continue
			}
			used = append(used, a)
			return a
		}
	}
	for _, v := range D.Labels {
		newindex := NewIndex()
		tmp[newindex] = v
	}
	rt := D.Labels
	D.Labels = tmp
	tmp = rt //now tmp has the old values, so it can be used as tmp for the next call to the function
	return D, used, tmp
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
	if l < g && (len(twotails) > 0 && twotails[0]) {
		p = float64(l)
	}
	return p / float64(len(nulls))
}

// This is an implementation of Altmann et al. Permutation Importance method.
// If you use this function, please cite:
// Altmann, André, Laura Toloşi, Oliver Sander, and Thomas Lengauer. "Permutation importance:
// a corrected feature importance measure." Bioinformatics 26, no. 10 (2010): 1340-1347.
// https://doi.org/10.1093/bioinformatics/btq13
func PermutationImportance(xgb *boo.MultiClass, D *utils.DataBunch, features *IDOrKey, npermut ...int) (float64, error) {
	//I'm so not calling this 'pimp'
	nperms := 1000
	if len(npermut) > 0 && npermut[0] > 0 {
		nperms = npermut[0]
	}
	score, _, _, err := VariableImportance(xgb, D, features)
	if err != nil {
		return -1, err
	}
	nulls := make([]float64, 0, nperms)
	var used, tmp []int
	for i := 0; i < nperms; i++ {
		D, used, tmp = PermuteLabels(D, used, tmp)
		ns, _, _, err := VariableImportance(xgb, D, features)
		if err != nil {
			return -1, err
		}
		nulls = append(nulls, ns)
	}
	//NOTE: Add parametric versions (the reference has normal, log-normal and gamma, in addition to the non-parametric method)
	return nonparam(score, nulls), nil
}
