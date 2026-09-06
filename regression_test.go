package boo

// Regression tests
// Whitebox on purpose: several checks need access to MultiClass's
// unexported fields (b, xgb, regression) to verify internal state that
// has no exported accessor, which requires living in package boo itself.

import (
	"bufio"
	"bytes"
	"math"
	"reflect"
	"sort"
	"sync"
	"testing"

	"github.com/rmera/boo/utils"
	"gonum.org/v1/gonum/mat"
)

// A small, cleanly 3-way-separable dataset, used where classes converging
// at different speeds matters (early-stop tests).
func separableDataBunch() *utils.DataBunch {
	data := [][]float64{}
	labels := []int{}
	for i := 0; i < 60; i++ {
		cls := i % 3
		var f float64
		switch cls {
		case 0:
			f = 0.0 // easiest, separates first
		case 1:
			f = 10.0
		case 2:
			f = 10.5 // closest to class 1, separates last
		}
		data = append(data, []float64{f, f + float64(i%2)})
		labels = append(labels, cls)
	}
	return &utils.DataBunch{Data: data, Labels: labels}
}

func trainSeparableModel(nrounds, earlyStop int, xgb bool) *MultiClass {
	D := separableDataBunch()
	O := DefaultOptions()
	O.XGB = xgb
	O.Rounds = nrounds
	O.EarlyStop = earlyStop
	O.LearningRate = 0.5
	return NewMultiClass(D, O)
}

func roundTrip(t *testing.T, m *MultiClass, opts ...*Options) *MultiClass {
	t.Helper()
	var buf bytes.Buffer
	w := bufio.NewWriter(&buf)
	if err := JSONMultiClass(m, "", w, opts...); err != nil {
		t.Fatalf("marshal error: %v", err)
	}
	w.Flush()
	raw := append([]byte{}, buf.Bytes()...)
	r := bufio.NewReader(bytes.NewReader(raw))
	m2, err := UnJSONMultiClass(r, opts...)
	if err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	return m2
}

// Tests that things work even for a small sample.
func TestSubSamplet(t *testing.T) {
	D := &utils.DataBunch{
		Data: [][]float64{
			{1, 2}, {2, 3}, {3, 4}, {4, 5}, {5, 6},
			{6, 7}, {7, 8}, {8, 9}, {9, 10}, {10, 11},
		},
		Labels: []int{0, 1, 0, 1, 0, 1, 0, 1, 0, 1},
	}
	O := DefaultOptions()
	O.XGB = false
	O.Rounds = 5
	m := NewMultiClass(D, O)
	if len(m.b) == 0 {
		t.Errorf("KNOWN BUG (xgb.go MinSample gate, still open): non-XGB gradient boosting trained 0 rounds, want %d", O.Rounds)
	}
}

func TestEarlyStop(t *testing.T) {
	m := trainSeparableModel(200, 1, true)
	if len(m.b) == 0 {
		t.Fatalf("setup failed: no rounds trained")
	}
	nclasses := m.Classes()
	for r, ensemble := range m.b {
		if len(ensemble) != nclasses {
			t.Fatalf("round %d: ensemble has %d slots, want a fixed %d (one per class, nil for stopped ones) -- compaction regression", r, len(ensemble), nclasses)
		}
	}
	// A prediction that doesn't panic and produces one probability per
	// class is a reasonable end-to-end sanity check on top of the
	// structural check above.
	p := m.PredictSingle([]float64{0, 1})
	if len(p) != nclasses {
		t.Fatalf("PredictSingle returned %d values, want %d", len(p), nclasses)
	}
}

// Read
func TestClassesReturnsClassCount(t *testing.T) {
	D := separableDataBunch()
	O := DefaultOptions()
	O.XGB = true
	O.EarlyStop = 0
	O.Rounds = 7
	m := NewMultiClass(D, O)
	if got, want := m.Classes(), 3; got != want {
		t.Errorf("Classes() = %d, want %d (there are 3 distinct labels; %d training rounds happened separately)", got, want, O.Rounds)
	}
	if got, want := len(m.b), O.Rounds; got != want {
		t.Fatalf("test invariant broken: len(m.b) = %d, want %d rounds", got, want)
	}
}

func TestRoundsCountsPerClassNotPerRound(t *testing.T) {
	m := trainSeparableModel(200, 1, true)
	for c := 0; c < m.Classes(); c++ {
		n := m.Rounds(c)
		if n <= 0 || n > len(m.b) {
			t.Errorf("Rounds(%d) = %d, want a value in (0, %d]", c, n, len(m.b))
		}
	}
	if got := m.Rounds(m.Classes()); got != -1 {
		t.Errorf("Rounds(%d) (out of range) = %d, want -1", m.Classes(), got)
	}
}

func TestClassLabelsRoundTrip(t *testing.T) {
	D := &utils.DataBunch{
		Data:   [][]float64{{70, 0}, {30, 1}, {90, 2}, {71, 3}, {31, 4}, {91, 5}},
		Labels: []int{7, 3, 9, 7, 3, 9}, // deliberately non-sequential labels
	}
	O := DefaultOptions()
	O.XGB = true
	O.Rounds = 5
	O.EarlyStop = 0
	m := NewMultiClass(D, O)

	m2 := roundTrip(t, m)
	if !reflect.DeepEqual(m.ClassLabels(), m2.ClassLabels()) {
		t.Fatalf("classLabels mismatch after round-trip: got %v want %v", m2.ClassLabels(), m.ClassLabels())
	}
}

func TestXGBAndRegressionRoundTripWithOptions(t *testing.T) {
	D := &utils.DataBunch{
		Data:   [][]float64{{0, 1}, {10, 11}, {20, 21}, {1, 2}, {11, 12}, {21, 22}},
		Labels: []int{0, 1, 2, 0, 1, 2},
	}
	O := DefaultOptions()
	O.XGB = true
	O.Rounds = 3
	m := NewMultiClass(D, O)

	m2 := roundTrip(t, m, O)
	if m2.xgb != m.xgb {
		t.Errorf("xgb mismatch after round-trip: got %v want %v", m2.xgb, m.xgb)
	}
	if m2.regression != m.regression {
		t.Errorf("regression mismatch after round-trip: got %v want %v", m2.regression, m.regression)
	}
}

func TestJSONOldFormatFileDoesNotPanic(t *testing.T) {
	D := separableDataBunch()
	O := DefaultOptions()
	O.Rounds = 2
	m := NewMultiClass(D, O)

	m2 := roundTrip(t, m) // no Options passed, either side
	if !reflect.DeepEqual(m.ClassLabels(), m2.ClassLabels()) {
		t.Fatalf("classLabels mismatch: %v vs %v", m.ClassLabels(), m2.ClassLabels())
	}
	t.Logf("old-format round trip OK (xgb=%v regression=%v, both expected false: not recoverable from a file with no embedded Options)", m2.xgb, m2.regression)
}

func TestNilTreeSlots_MarshalUnmarshalDoNotPanic(t *testing.T) {
	m := trainSeparableModel(200, 1, true)
	foundNil := false
	for _, ensemble := range m.b {
		for _, tree := range ensemble {
			if tree == nil {
				foundNil = true
			}
		}
	}
	if !foundNil {
		t.Fatalf("test setup failed: no nil tree slot present, can't exercise the NIL-marker path")
	}
	m2 := roundTrip(t, m) // must not panic on write or read

	if len(m.b) != len(m2.b) {
		t.Fatalf("round count mismatch: orig=%d reloaded=%d", len(m.b), len(m2.b))
	}
	mismatch := 0
	for i := range m.b {
		if len(m.b[i]) != len(m2.b[i]) {
			t.Fatalf("round %d: class-slot count mismatch: %d vs %d", i, len(m.b[i]), len(m2.b[i]))
		}
		for k := range m.b[i] {
			origNil := m.b[i][k] == nil
			gotNil := m2.b[i][k] == nil
			if origNil != gotNil {
				mismatch++
				t.Errorf("round %d class %d: nil-ness mismatch, orig nil=%v reloaded nil=%v", i, k, origNil, gotNil)
			}
		}
	}
	if mismatch == 0 {
		t.Logf("nil/non-nil pattern preserved exactly across all %d rounds", len(m2.b))
	}

	// Predictions from the reloaded model should work too.
	p := m2.PredictSingle([]float64{0, 1})
	t.Logf("reloaded model prediction: %v", p)
}

func TestLastRoundNotDroppedOnReload(t *testing.T) {
	// Big enough that XGB's default 0.8 row subsampling doesn't dip below
	// MinSample and starve every round via the still-open bug 1.
	D := separableDataBunch()
	O := DefaultOptions()
	O.Rounds = 5
	O.EarlyStop = 0
	m := NewMultiClass(D, O)
	if len(m.b) != O.Rounds {
		t.Fatalf("test invariant broken: len(m.b) = %d, want %d rounds", len(m.b), O.Rounds)
	}

	m2 := roundTrip(t, m)
	if len(m.b) != len(m2.b) {
		t.Fatalf("last round lost on reload: orig=%d rounds, reloaded=%d rounds", len(m.b), len(m2.b))
	}
}

func TestLastRoundNotDroppedOnReload_WithEarlyStopAndNilSlots(t *testing.T) {
	m := trainSeparableModel(200, 1, true)
	m2 := roundTrip(t, m)
	if len(m.b) != len(m2.b) {
		t.Fatalf("last round lost on reload: orig=%d rounds, reloaded=%d rounds", len(m.b), len(m2.b))
	}
}

func TestZeroRoundModelRoundTrip(t *testing.T) {
	D := &utils.DataBunch{Data: [][]float64{{0, 1}, {1, 0}}, Labels: []int{0, 1}}
	O := DefaultOptions()
	O.Rounds = 0
	m := NewMultiClass(D, O)
	m2 := roundTrip(t, m)
	if len(m.b) != len(m2.b) {
		t.Fatalf("zero-round model: orig=%d reloaded=%d", len(m.b), len(m2.b))
	}
}

func TestRegressionSetterActuallySetsFlag(t *testing.T) {
	O := DefaultOptions()
	if O.Regression() {
		t.Fatalf("setup: expected O.Regression() to start false")
	}
	O.Regression(true)
	if !O.Regression() {
		t.Errorf("O.Regression(true) did not set the flag")
	}
}

func TestBug5_RegressionModeReachableViaPublicAPI(t *testing.T) {
	D := &utils.DataBunch{
		Data:        [][]float64{{0}, {1}, {2}, {3}, {4}},
		FloatLabels: []float64{0.1, 1.1, 2.1, 3.1, 4.1},
	}
	O := DefaultOptions()
	O.Regression(true)
	O.Rounds = 3
	m := NewMultiClass(D, O)
	if !m.regression {
		t.Errorf("trained model is not in regression mode despite O.Regression(true)")
	}
}

func TestEqualComparesTreeMethodToEachOther(t *testing.T) {
	a := DefaultOptions()
	b := a.Clone()
	a.TreeMethod = "hist"
	b.TreeMethod = "hist"
	if !a.Equal(b) {
		t.Errorf("Equal() returned false for two Options with the same non-'exact' TreeMethod (%q)", a.TreeMethod)
	}

	c := a.Clone()
	c.TreeMethod = "exact" // genuinely different from a's "hist"
	if a.Equal(c) {
		t.Errorf("Equal() returned true for genuinely different TreeMethod values (%q vs %q)", a.TreeMethod, c.TreeMethod)
	}
}

func TestTreeFeatureNameDoesNotPanicOnShortSlice(t *testing.T) {
	tr := &Tree{splitFeatureIndex: 5}
	got := tr.feature([]string{"a", "b"}) // too short: index 5 out of range
	want := " 5"
	if got != want {
		t.Errorf("feature() = %q, want %q (fallback to numeric index)", got, want)
	}
}

func TestFeatsAddIsConcurrencySafe(t *testing.T) {
	f := NewFeats(true)
	var wg sync.WaitGroup
	const goroutines = 50
	const perGoroutine = 200
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < perGoroutine; i++ {
				f.Add(i%10, 1.0) // heavy overlap: only 10 distinct features
			}
		}()
	}
	wg.Wait()

	if got, want := len(f.feat), 10; got != want {
		t.Errorf("got %d distinct features, want %d -- duplicate entries from a check-then-act race", got, want)
	}
	total := 0.0
	for _, g := range f.gains {
		total += g
	}
	wantTotal := float64(goroutines * perGoroutine)
	if total != wantTotal {
		t.Errorf("total gain = %v, want %v -- lost updates from a race", total, wantTotal)
	}
}

func TestSubSampleMonotonicity(t *testing.T) {
	totdata := 10000
	probs := []float64{0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0}
	prevFrac := -1.0
	for _, p := range probs {
		n := len(SubSample(totdata, p))
		frac := float64(n) / float64(totdata)
		t.Logf("subsample=%.2f -> sampled %d/%d (%.3f%% of rows)", p, n, totdata, frac*100)
		if frac < prevFrac {
			t.Errorf("non-monotonic: subsample=%.2f produced a smaller fraction (%.4f) than a lower subsample value did (%.4f)", p, frac, prevFrac)
		}
		// fraction sampled should be close to the probability itself
		if diff := frac - p; diff > 0.03 || diff < -0.03 {
			t.Errorf("subsample=%.2f: sampled fraction %.4f is not close to the requested probability", p, frac)
		}
		prevFrac = frac
	}
}

func TestFeatsFeatsReturnsTopN(t *testing.T) {
	f := NewFeats(true) // xgb: higher gain is "better", sorts first
	f.Add(3, 5.0)
	f.Add(1, 9.0)
	f.Add(2, 1.0)
	sort.Sort(f)

	if got, want := f.Feats(2), []int{1, 3}; !reflect.DeepEqual(got, want) {
		t.Errorf("Feats(2) = %v, want %v (highest-gain features first for xgb)", got, want)
	}
	if got, want := f.Feats(0), []int{1, 3, 2}; !reflect.DeepEqual(got, want) {
		t.Errorf("Feats(0) = %v, want %v (0 means \"all\")", got, want)
	}
	if got, want := f.Feats(10), []int{1, 3, 2}; !reflect.DeepEqual(got, want) {
		t.Errorf("Feats(10) (more than available) = %v, want %v (just what we have)", got, want)
	}
}

func TestFeatsMergeSumsOverlappingGains(t *testing.T) {
	f1 := NewFeats(false)
	f1.Add(0, 1.0)
	f1.Add(1, 2.0)
	f2 := NewFeats(false)
	f2.Add(1, 3.0) // overlaps with f1's feature 1
	f2.Add(2, 4.0)

	f1.Merge(f2)

	want := map[int]float64{0: 1.0, 1: 5.0, 2: 4.0}
	if got := len(f1.feat); got != len(want) {
		t.Fatalf("after merge, got %d distinct features, want %d", got, len(want))
	}
	for i, feat := range f1.feat {
		if g, ok := want[feat]; !ok || g != f1.gains[i] {
			t.Errorf("feature %d: gain = %v, want %v", feat, f1.gains[i], want[feat])
		}
	}
}

func TestClassesFromProbsPicksArgmax(t *testing.T) {
	p := mat.NewDense(2, 3, []float64{
		0.1, 0.7, 0.2,
		0.5, 0.2, 0.3,
	})
	classes := ClassesFromProbs(p)
	if got, want := classes.At(0, 0), 1.0; got != want {
		t.Errorf("row 0: predicted class %v, want %v", got, want)
	}
	if got, want := classes.At(1, 0), 0.0; got != want {
		t.Errorf("row 1: predicted class %v, want %v", got, want)
	}
}

func TestLogOddsFromProbsMatchesFormula(t *testing.T) {
	// row = [0.5, 0.5]: s = 1.0, so for each element v, log(v*(s-v)) = log(0.25).
	p := mat.NewDense(1, 2, []float64{0.5, 0.5})
	got := LogOddsFromProbs(p)
	want := math.Log(0.25)
	for j := 0; j < 2; j++ {
		if v := got.At(0, j); math.Abs(v-want) > 1e-9 {
			t.Errorf("LogOddsFromProbs()[0][%d] = %v, want %v", j, v, want)
		}
	}
}

func TestTreeBranchesCountsAllNodes(t *testing.T) {
	o := DefaultGTreeOptions()
	o.MaxDepth = 1
	o.MinChildWeight = 1
	data := [][]float64{{0}, {0}, {10}, {10}}
	o.Y = []float64{0, 0, 1, 1}
	tree := NewTree(data, o)

	if tree.Leaf() {
		t.Fatalf("setup failed: root is a leaf, expected one split given the separable data")
	}
	if !tree.left.Leaf() || !tree.right.Leaf() {
		t.Fatalf("setup failed: children should be leaves at MaxDepth 1")
	}
	if got, want := tree.Branches(), 3; got != want {
		t.Errorf("Branches() = %d, want %d (root + 2 leaves)", got, want)
	}
}

func TestProbabilitiesNilForRegression(t *testing.T) {
	D := &utils.DataBunch{
		Data:        [][]float64{{0}, {1}, {2}, {3}, {4}},
		FloatLabels: []float64{0.1, 1.1, 2.1, 3.1, 4.1},
	}
	O := DefaultOptions()
	O.Regression(true)
	O.Rounds = 3
	m := NewMultiClass(D, O)

	r, w := m.Probabilities(D)
	if r != nil || w != nil {
		t.Errorf("Probabilities() on a regression model = (%v, %v), want (nil, nil): not well-defined for regression", r, w)
	}
}

func TestProbabilitiesShapeForClassification(t *testing.T) {
	m := trainSeparableModel(50, 0, true)
	D := separableDataBunch()
	right, wrong := m.Probabilities(D)
	if right != nil && len(right) != 2 {
		t.Errorf("Probabilities() right = %v, want a [mean, stddev] pair (len 2) or nil", right)
	}
	if wrong != nil && len(wrong) != 2 {
		t.Errorf("Probabilities() wrong = %v, want a [mean, stddev] pair (len 2) or nil", wrong)
	}
}

func TestAccuracyRegressionReturnsInverseRMSD(t *testing.T) {
	D := &utils.DataBunch{
		Data:        [][]float64{{0}, {1}, {2}, {3}, {4}, {5}, {6}, {7}},
		FloatLabels: []float64{0.1, 1.3, 1.9, 3.2, 4.1, 4.8, 6.3, 6.9},
	}
	O := DefaultOptions()
	O.Regression(true)
	O.Rounds = 10
	O.EarlyStop = 0
	m := NewMultiClass(D, O)

	acc := m.Accuracy(D)
	if math.IsNaN(acc) {
		t.Fatalf("Accuracy() on a regression model returned NaN")
	}
	if acc <= 0 {
		t.Errorf("Accuracy() (1/RMSD) = %v, want a positive value", acc)
	}
}

func TestRoundsWithNoArgumentDefaultsToClassZero(t *testing.T) {
	m := trainSeparableModel(200, 1, true)
	if got, want := m.Rounds(), m.Rounds(0); got != want {
		t.Errorf("Rounds() = %d, want Rounds(0) = %d", got, want)
	}
}

func TestNewMultiClassFillsMissingLossAndActivation(t *testing.T) {
	D := separableDataBunch()
	O := &Options{
		Rounds:         5,
		MaxDepth:       3,
		LearningRate:   0.3,
		MinChildWeight: 1,
	} // Loss and Activation deliberately left nil
	m := NewMultiClass(D, O)

	if m.activation == nil {
		t.Fatalf("NewMultiClass left activation nil instead of defaulting it")
	}
	p := m.PredictSingle([]float64{0, 1})
	if len(p) != m.Classes() {
		t.Errorf("PredictSingle with defaulted activation returned %d values, want %d", len(p), m.Classes())
	}
}
