package utils

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"
)

// Returns a copy of D with the Labels randomly permuted (shuffled).
// D.Labels must be filled (len>0). If FloatLabels is filled, its lenght must
// match that of D.Labels
func PermuteLabels(D *DataBunch) *DataBunch {
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

// Returns a copy of D with the features in the given index set scrambled. All features in the given
// set are scrambled randomly but identically. i.e. using the same random permutation for both each time.
// by 'scrambled' I mean that the values of each chosen feature are scramled among vectors.
func PermuteFeatures(D *DataBunch, features *IDOrKey) (*DataBunch, error) {
	feats, err := features.Get(D, true)
	if err != nil {
		return nil, fmt.Errorf("ScrambleFeature: Failed to obtain feature index: %w", err)
	}
	ret := D.Copy()
	samples := len(D.Data)
	newindexes := rand.Perm(samples)
	for i, v := range D.Data {
		for _, feat := range feats {
			ret.Data[newindexes[i]][feat] = v[feat]
		}
	}
	return ret, nil
}

// Contains either the indexes/IDs or the keys of a number of features
// basically, a group of features.
// It will return all the features of the group at once.

type IDOrKey struct {
	IDs    []int
	Keys   []string
	NCKeys []string
}

// note that if some of  the I.Keys are missing from ID
func (I *IDOrKey) Get(D *DataBunch, nocaps ...bool) ([]int, error) {
	//the c ontent of these 2 will depend if we are using cap sensitivity or not.
	keys := I.Keys
	proc := func(s string) string { return s }
	if len(I.Keys) > 0 {
		if len(nocaps) > 0 && nocaps[0] {
			if I.NCKeys == nil {
				I.NCKeys = make([]string, len(I.Keys))
				for i, v := range I.Keys {
					I.NCKeys[i] = strings.ToLower(v)
				}
			}
			keys = I.NCKeys
			proc = strings.ToLower
		}
		ret := make([]int, 0, len(I.Keys))

		for i, v := range D.Keys {

			if slices.Contains(keys, proc(v)) {
				ret = append(ret, i)
			}
		}
		if len(ret) == 0 {
			return I.IDs, fmt.Errorf("IDOrKey.Get: Keys in  %v are not found in DataBunch", I.Keys)
		}
		return ret, nil
	}
	//if we have no keys, we try with IDs
	if len(I.IDs) > 0 {
		return I.IDs, nil
	}
	return nil, fmt.Errorf("IDOrKey.Get: I contain neither keys nor IDs")

}
