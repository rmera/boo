package an

import (
	"fmt"
	"math/rand/v2"
	"slices"
	"strings"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

// Returns a copy of D with the features in the given index set scrambled. All features in the given
// set are scrambled randomly but identically. i.e. using the same random permutation for both each time.
// by 'scrambled' I mean that the values of each chosen feature are scramled among vectors.
func PermuteFeatures(D *utils.DataBunch, features *IDOrKey) (*utils.DataBunch, error) {
	feats, err := features.Get(D, true)
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

// Contains either the indexes/IDs or the keys of a number of features
// basically, a group of features.
// It will return all the features of the group at once.

type IDOrKey struct {
	IDs    []int
	Keys   []string
	NCKeys []string
}

// note that if some of  the I.Keys are missing from ID
func (I *IDOrKey) Get(D *utils.DataBunch, nocaps ...bool) ([]int, error) {
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

func VariableImportance(xgb *boo.MultiClass, D *utils.DataBunch, feature *IDOrKey) ([3]float64, error) {
	IniAcc := xgb.Accuracy(D)
	scdata := D
	var err error
	scdata, err = PermuteFeatures(scdata, feature)
	if err != nil {
		return [3]float64{0, 0, 0}, fmt.Errorf("VariableImportance: Couldn't do permutation: %w", err)
	}

	/*
		fmt.Println("ori", D)      ///////////////////////////////////
		fmt.Println("mod", scdata) /////////////////
		fg, _ := feature.Get(D)    /////////////////////////
		fmt.Println("feature", fg) //////////////////
	*/
	ScAcc := xgb.Accuracy(scdata)
	return [3]float64{IniAcc - ScAcc, IniAcc, ScAcc}, nil
}
