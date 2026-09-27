package an

import (
	"fmt"

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
