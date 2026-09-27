package utils

import (
	"slices"
	"testing"
)

func scrambleTestDataBunch() *DataBunch {
	return &DataBunch{
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
	D := scrambleTestDataBunch()
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
	D := scrambleTestDataBunch()
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
	D := scrambleTestDataBunch()
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
	D := scrambleTestDataBunch()
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
	D := scrambleTestDataBunch()
	ik := &IDOrKey{Keys: []string{"nope"}}
	_, err := ik.Get(D, true)
	if err == nil {
		t.Errorf("IDOrKey.Get: expected an error for missing keys, got nil")
	}
}

func TestIDOrKeyGetNeitherKeysNorIDs(t *testing.T) {
	D := scrambleTestDataBunch()
	ik := &IDOrKey{}
	_, err := ik.Get(D)
	if err == nil {
		t.Errorf("IDOrKey.Get: expected an error when neither Keys nor IDs are set, got nil")
	}
}
