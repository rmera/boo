package an

import (
	"fmt"
	"math/rand/v2"
	"slices"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

// Returns a copy of D with the features in the given index set scrambled. All features in the given
// set are scrambled randomly but identically. i.e. using the same random permutation for both each time.
func PermuteFeatures(D *utils.DataBunch, features *IDOrKey) (*utils.DataBunch, error) {
	feats, err := features.Get(D)
	if err != nil {
		return nil, fmt.Errorf("ScrambleFeature: Failed to obtain feature index: %w", err)
	}
	ret := D.Copy()
	samples := len(D.Data)

	used := make([]int, 0, 10)
	NewIndex := func() int {
		for {
			a := rand.IntN(samples)
			if slices.Contains(used, a) {
				continue
			}
			used = append(used, a)
			return a
		}
		return -1
	}
	for _, v := range D.Data {
		newindex := NewIndex()
		for _, feat := range feats {
			ret.Data[newindex][feat] = v[feat]
		}
	}
	return ret, nil
}

// Contains either the indexes/IDs or the keys of a number of featuree
type IDOrKey struct {
	IDs  []int
	Keys []string
}

func (I *IDOrKey) Get(D ...*utils.DataBunch) ([]int, error) {
	if len(D) == 0 {
		return I.IDs, nil
	}
	if D[0] == nil {
		return I.IDs, fmt.Errorf("IDOrKey.Get: Given a nil DataBunch")
	}
	ret := make([]int, 0, len(I.Keys))
	for i, v := range D[0].Keys {
		if slices.Contains(I.Keys, v) {
			ret = append(ret, i)
		}
	}
	if len(ret) == 0 {
		return I.IDs, fmt.Errorf("Keys in  %v are not found in DataBunch", I.Keys)
	}
	return ret, nil
}

func VariableImportance(xgb *boo.MultiClass, D *utils.DataBunch, feature *IDOrKey, nperm ...int) (float64, float64, float64, error) {
	n := 1 //default value
	if len(nperm) > 0 {
		n = nperm[0]
	}
	scdata := D
	var err error
	for i := 0; i < n; i++ {
		scdata, err = PermuteFeatures(scdata, feature)
		if err != nil {
			return 0, 0, 0, fmt.Errorf("PermutationImportance: Couldn't do permutation: %w", err)
		}
	}
	IniAcc := xgb.Accuracy(D)
	ScAcc := xgb.Accuracy(scdata)
	return IniAcc - ScAcc, IniAcc, ScAcc, nil
}
