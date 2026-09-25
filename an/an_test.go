package an

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

func floatsClose(a, b, tol float64) bool {
	return math.Abs(a-b) <= tol
}

func TestStability(t *testing.T) {
	Z := [][]float64{{1, 1, 0, 0, 1, 1, 0, 0, 1}, {1, 1, 1, 0, 0, 1, 1, 0, 0}, {0, 1, 1, 1, 0, 1, 1, 1, 0}, {0, 1, 1, 0, 0, 1, 1, 0, 1}, {1, 0, 1, 1, 0, 0, 0, 0, 1}}
	stab, vari := StabilityAndVariance(Z)
	wantStab := 0.009999999999999898
	wantVar := 0.014128020000000002
	if !floatsClose(stab, wantStab, 1e-9) {
		t.Errorf("StabilityAndVariance: stability got %v, want %v", stab, wantStab)
	}
	if !floatsClose(vari, wantVar, 1e-9) {
		t.Errorf("StabilityAndVariance: variance got %v, want %v", vari, wantVar)
	}
	if fs := FeatStability(Z); !floatsClose(fs, wantStab, 1e-9) {
		t.Errorf("FeatStability: got %v, want %v", fs, wantStab)
	}
}

func TestGrouping(t *testing.T) {
	group := [][]int{{1, 2}, {3, 4}, {5, 6}}
	input := []int{1, 2, 5, 6}

	f := MakeGrouperFunc(group)

	got, n := f(input)
	want := []int{0, 2}
	if n != 3 {
		t.Errorf("MakeGrouperFunc: group count got %d, want 3", n)
	}
	if !slices.Equal(got, want) {
		t.Errorf("MakeGrouperFunc: got %v, want %v", got, want)
	}
}

func TestMakeGrouperFuncEmptyInput(t *testing.T) {
	group := [][]int{{1, 2}, {3, 4}}
	f := MakeGrouperFunc(group)
	got, n := f(nil)
	if n != 2 {
		t.Errorf("MakeGrouperFunc: group count got %d, want 2", n)
	}
	if len(got) != 0 {
		t.Errorf("MakeGrouperFunc: expected no groups represented, got %v", got)
	}
}

func TestNonparamNoTies(t *testing.T) {
	nulls := []float64{0.1, 0.2, 0.3, 0.4, 0.5}
	// 0.4 and 0.5 are the only nulls greater than 0.35. With no ties, both
	// modes must agree.
	want := 2.0 / 5.0
	if p := nonparam(0.35, nulls); !floatsClose(p, want, 1e-12) {
		t.Errorf("nonparam: got %v, want %v", p, want)
	}
	if p := nonparam(0.35, nulls, true); !floatsClose(p, want, 1e-12) {
		t.Errorf("nonparam strict: got %v, want %v", p, want)
	}
}

func TestNonparamTies(t *testing.T) {
	nulls := []float64{0.1, 0.2, 0.3, 0.3, 0.5}
	// Default: nulls at least as extreme as 0.3 are 0.3, 0.3 and 0.5.
	if p, want := nonparam(0.3, nulls), 3.0/5.0; !floatsClose(p, want, 1e-12) {
		t.Errorf("nonparam: got %v, want %v", p, want)
	}
	// Strict: only 0.5 is more extreme than 0.3.
	if p, want := nonparam(0.3, nulls, true), 1.0/5.0; !floatsClose(p, want, 1e-12) {
		t.Errorf("nonparam strict: got %v, want %v", p, want)
	}
	// An explicit false must behave like the default.
	if p, want := nonparam(0.3, nulls, false), 3.0/5.0; !floatsClose(p, want, 1e-12) {
		t.Errorf("nonparam with false: got %v, want %v", p, want)
	}
}

func TestNonparamDoesNotModifyNulls(t *testing.T) {
	nulls := []float64{0.5, 0.1, 0.4, 0.2}
	orig := slices.Clone(nulls)
	nonparam(0.3, nulls)
	if !slices.Equal(nulls, orig) {
		t.Errorf("nonparam modified its input: got %v, want %v", nulls, orig)
	}
}

// Compares nonparam against the non-parametric branch of the reference R
// implementation by Altmann et al. (https://github.com/andrealtmann/PIMP, PIMP.R):
//
//	pv <- sum(rnd[,i]>=imp[i])/length(rnd[,i])
//
// The expected values were obtained by evaluating that expression on each case.
func TestNonparamMatchesReference(t *testing.T) {
	cases := []struct {
		imp   float64
		nulls []float64
		want  float64
	}{
		{0.35, []float64{0.1, 0.2, 0.3, 0.4, 0.5}, 0.4},
		{0.3, []float64{0.1, 0.2, 0.3, 0.3, 0.5}, 0.6},
		// Typical for an uninformative feature: many importances exactly 0.
		{0.0, []float64{0.0, 0.0, -0.01, 0.02, 0.0, 0.0, -0.005, 0.0}, 0.75},
		{0.9, []float64{0.1, 0.2, 0.3}, 0.0},
		{-0.5, []float64{0.1, 0.2, 0.3}, 1.0},
		{0.05, []float64{0.05, 0.05, 0.05, 0.05}, 1.0},
	}
	for _, c := range cases {
		if p := nonparam(c.imp, c.nulls); !floatsClose(p, c.want, 1e-12) {
			t.Errorf("nonparam(%v, %v): got %v, reference gives %v", c.imp, c.nulls, p, c.want)
		}
	}
}

func testDataBunch() *utils.DataBunch {
	return &utils.DataBunch{
		Data: [][]float64{
			{1, 10, 100},
			{2, 20, 200},
			{3, 30, 300},
			{4, 40, 400},
			{5, 50, 500},
		},
		Keys:   []string{"a", "b", "c"},
		Labels: []int{0, 1, 0, 1, 0},
	}
}

func TestPermuteLabelsIsAPermutation(t *testing.T) {
	D := testDataBunch()
	orig := slices.Clone(D.Labels)
	D2 := PermuteLabels(D)

	if !slices.Equal(D.Labels, orig) {
		t.Errorf("PermuteLabels mutated the original DataBunch's Labels: got %v, want %v", D.Labels, orig)
	}
	if len(D2.Labels) != len(orig) {
		t.Fatalf("PermuteLabels: got %d labels, want %d", len(D2.Labels), len(orig))
	}
	sortedOrig := slices.Clone(orig)
	sortedNew := slices.Clone(D2.Labels)
	slices.Sort(sortedOrig)
	slices.Sort(sortedNew)
	if !slices.Equal(sortedOrig, sortedNew) {
		t.Errorf("PermuteLabels: result is not a permutation of the original: got %v, from %v", D2.Labels, orig)
	}
}

func TestPermuteFeaturesOnlyTouchesSelectedColumns(t *testing.T) {
	D := testDataBunch()
	origData := make([][]float64, len(D.Data))
	for i, row := range D.Data {
		origData[i] = slices.Clone(row)
	}

	features := &IDOrKey{Keys: []string{"a"}}
	D2, err := PermuteFeatures(D, features)
	if err != nil {
		t.Fatalf("PermuteFeatures: unexpected error: %v", err)
	}

	for i, row := range D.Data {
		if !slices.Equal(row, origData[i]) {
			t.Errorf("PermuteFeatures mutated the original DataBunch at row %d: got %v, want %v", i, row, origData[i])
		}
	}

	// Columns 1 and 2 were not selected for permutation, so they must be untouched.
	for i := range D2.Data {
		if D2.Data[i][1] != origData[i][1] || D2.Data[i][2] != origData[i][2] {
			t.Errorf("PermuteFeatures changed an unselected column at row %d: got %v, want cols 1,2 = %v,%v", i, D2.Data[i], origData[i][1], origData[i][2])
		}
	}

	// Column 0 (selected) should still contain the same values, just reordered.
	var origCol, newCol []float64
	for i := range D2.Data {
		origCol = append(origCol, origData[i][0])
		newCol = append(newCol, D2.Data[i][0])
	}
	slices.Sort(origCol)
	slices.Sort(newCol)
	if !slices.Equal(origCol, newCol) {
		t.Errorf("PermuteFeatures: selected column is not a permutation of the original: got %v, want a permutation of %v", newCol, origCol)
	}
}

func TestIDOrKeyGetByKeys(t *testing.T) {
	D := testDataBunch()
	ik := &IDOrKey{Keys: []string{"B", "c"}}
	got, err := ik.Get(D, true)
	if err != nil {
		t.Fatalf("IDOrKey.Get: unexpected error: %v", err)
	}
	want := []int{1, 2}
	if !slices.Equal(got, want) {
		t.Errorf("IDOrKey.Get by keys: got %v, want %v", got, want)
	}
}

func TestIDOrKeyGetByIDs(t *testing.T) {
	D := testDataBunch()
	ik := &IDOrKey{IDs: []int{0, 2}}
	got, err := ik.Get(D)
	if err != nil {
		t.Fatalf("IDOrKey.Get: unexpected error: %v", err)
	}
	if !slices.Equal(got, []int{0, 2}) {
		t.Errorf("IDOrKey.Get by IDs: got %v, want %v", got, []int{0, 2})
	}
}

func TestIDOrKeyGetMissingKeys(t *testing.T) {
	D := testDataBunch()
	ik := &IDOrKey{Keys: []string{"nope"}}
	_, err := ik.Get(D, true)
	if err == nil {
		t.Errorf("IDOrKey.Get: expected an error for missing keys, got nil")
	}
}

func TestIDOrKeyGetNeitherKeysNorIDs(t *testing.T) {
	D := testDataBunch()
	ik := &IDOrKey{}
	_, err := ik.Get(D)
	if err == nil {
		t.Errorf("IDOrKey.Get: expected an error when neither Keys nor IDs are set, got nil")
	}
}

func trainTestModel(t *testing.T) (*boo.MultiClass, *utils.DataBunch) {
	t.Helper()
	D := &utils.DataBunch{
		Data: [][]float64{
			{0, 1}, {10, 11}, {20, 21}, {1, 0}, {11, 10}, {21, 20},
			{0, 2}, {10, 12}, {20, 22}, {2, 0}, {12, 10}, {22, 20},
		},
		Keys:   []string{"f0", "f1"},
		Labels: []int{0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2},
	}
	O := boo.DefaultXOptions()
	O.Rounds = 5
	xgb := boo.NewMultiClass(D, O)
	return xgb, D
}

func TestVariableImportance(t *testing.T) {
	xgb, D := trainTestModel(t)
	res, err := VariableImportance(xgb, D, &IDOrKey{Keys: []string{"f0"}})
	if err != nil {
		t.Fatalf("VariableImportance: unexpected error: %v", err)
	}
	wantIni := xgb.Accuracy(D)
	if res[1] != wantIni {
		t.Errorf("VariableImportance: initial accuracy got %v, want %v", res[1], wantIni)
	}
	if !floatsClose(res[0], res[1]-res[2], 1e-12) {
		t.Errorf("VariableImportance: importance (%v) should equal IniAcc-ScAcc (%v-%v)", res[0], res[1], res[2])
	}
}

func TestVariableImportanceUnknownKeyErrors(t *testing.T) {
	xgb, D := trainTestModel(t)
	_, err := VariableImportance(xgb, D, &IDOrKey{Keys: []string{"nope"}})
	if err == nil {
		t.Errorf("VariableImportance: expected an error for an unknown feature key, got nil")
	}
}

func TestPermutationImportance(t *testing.T) {
	_, D := trainTestModel(t)
	o := DefaultPermImportanceOptions()
	o.BooOpts = boo.DefaultXOptions()
	o.BooOpts.Rounds = 5
	o.LabelPerms = 20
	vi, p, err := PermutationImportance(D, D, &IDOrKey{Keys: []string{"f0"}}, o)
	if err != nil {
		t.Fatalf("PermutationImportance: unexpected error: %v", err)
	}
	if math.IsNaN(vi) || vi < -100 || vi > 100 {
		t.Errorf("PermutationImportance: importance (a difference of accuracy percentages) out of range [-100,100]: %v", vi)
	}
	if p < 0 || p > 1 {
		t.Errorf("PermutationImportance: p-value out of range [0,1]: %v", p)
	}
}

func TestPermutationImportanceNoBooOptions(t *testing.T) {
	_, D := trainTestModel(t)
	o := DefaultPermImportanceOptions()
	_, _, err := PermutationImportance(D, D, &IDOrKey{Keys: []string{"f0"}}, o)
	if err == nil {
		t.Errorf("PermutationImportance: expected an error when BooOpts is nil, got nil")
	}
}

func TestStabilityOnDataVar(t *testing.T) {
	_, D := trainTestModel(t)
	O := boo.DefaultXOptions()
	O.Rounds = 3
	stab, vari := StabilityOnDataVar(D, O, 4, 1)
	if math.IsNaN(stab) || math.IsNaN(vari) {
		t.Errorf("StabilityOnDataVar: got NaN result: stability=%v, variance=%v", stab, vari)
	}
	if stab < -1 || stab > 1 {
		t.Errorf("StabilityOnDataVar: stability out of expected range [-1,1]: %v", stab)
	}
}

// A toy binary problem: f0 is strongly informative, f1 weakly informative,
// f2 and f3 are pure noise. The generator is seeded, so the data is fixed.
func pimpToyData(n int, seed uint64) *utils.DataBunch {
	r := rand.New(rand.NewPCG(seed, seed+1))
	D := &utils.DataBunch{Keys: []string{"f0", "f1", "f2", "f3"}}
	for i := 0; i < n; i++ {
		x := []float64{r.Float64(), r.Float64(), r.Float64(), r.Float64()}
		s := x[0] + 0.3*x[1] + 0.3*r.NormFloat64()
		l := 0
		if s > 0.65 {
			l = 1
		}
		D.Data = append(D.Data, x)
		D.Labels = append(D.Labels, l)
	}
	return D
}

// On this data (train seed 1, test seed 2), the reference PIMP pipeline (random forest,
// accuracy-drop VI on the test set, 50 label permutations, non-parametric p-values as in
// PIMP.R) gave VI=24.3, p=0.00 for f0 and VI=-0.85, p=0.58 for f3. The permutations make
// the results random, so only the clear-cut conclusions are checked.
func TestPermutationImportanceToy(t *testing.T) {
	if testing.Short() {
		t.Skip("slow: trains many models")
	}
	train := pimpToyData(200, 1)
	test := pimpToyData(200, 2)
	o := DefaultPermImportanceOptions()
	o.BooOpts = boo.DefaultXOptions()
	o.BooOpts.Rounds = 10
	o.LabelPerms = 20

	vi0, p0, err := PermutationImportance(train, test, &IDOrKey{Keys: []string{"f0"}}, o)
	if err != nil {
		t.Fatalf("PermutationImportance: unexpected error: %v", err)
	}
	if p0 >= 0.05 {
		t.Errorf("PermutationImportance: informative feature f0 should be significant, got VI=%v p=%v", vi0, p0)
	}
	vi3, p3, err := PermutationImportance(train, test, &IDOrKey{Keys: []string{"f3"}}, o)
	if err != nil {
		t.Fatalf("PermutationImportance: unexpected error: %v", err)
	}
	if vi3 >= vi0 {
		t.Errorf("PermutationImportance: noise feature f3 (VI=%v, p=%v) should be less important than f0 (VI=%v)", vi3, p3, vi0)
	}
}
