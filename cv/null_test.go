package cv

import (
	"math"
	"math/rand/v2"
	"slices"
	"testing"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
	"gonum.org/v1/gonum/stat"
)

// A binary problem where the label is determined by f0 alone; f1 is noise.
// The generator is seeded, so the data is fixed.
func nullToyData(n int, seed uint64) *utils.DataBunch {
	r := rand.New(rand.NewPCG(seed, seed+1))
	D := &utils.DataBunch{Keys: []string{"f0", "f1"}}
	for i := 0; i < n; i++ {
		x := []float64{r.Float64(), r.Float64()}
		l := 0
		if x[0] > 0.5 {
			l = 1
		}
		D.Data = append(D.Data, x)
		D.Labels = append(D.Labels, l)
	}
	return D
}

func nullTestOptions() *boo.Options {
	O := boo.DefaultXOptions()
	O.Rounds = 10
	O.MaxDepth = 2
	return O
}

// On data with a real signal, the accuracy obtained with permuted labels should be
// clearly lower than that obtained with the real labels. The original data must
// not be modified, one accuracy per permutation must be returned, and the first
// return value must be their mean.
func TestNullRepeatedCrossValidation(t *testing.T) {
	data := nullToyData(60, 1)
	origLabels := slices.Clone(data.Labels)
	O := nullTestOptions()

	accs, err := RepeatedCrossvalidation(data, 3, 3, &Options{O: O, Conc: false})
	if err != nil {
		t.Fatalf("RepeatedCrossvalidation returned an error: %v", err)
	}
	real := stat.Mean(accs, nil)

	nperm := 5
	null, nulls, err := NullRepeatedCrossValidation(data, 3, 3, nperm, &Options{O: O, Conc: false})
	if err != nil {
		t.Fatalf("NullRepeatedCrossValidation returned an error: %v", err)
	}
	t.Logf("accuracy with real labels: %.1f, with permuted labels: %.1f %v", real, null, nulls)
	if len(nulls) != nperm {
		t.Fatalf("got %d per-permutation accuracies, want %d", len(nulls), nperm)
	}
	for _, a := range nulls {
		if a < 0 || a > 100 {
			t.Errorf("per-permutation accuracy %v out of the expected [0,100] range", a)
		}
	}
	if math.Abs(null-stat.Mean(nulls, nil)) > 1e-9 {
		t.Errorf("null accuracy %v is not the mean of the per-permutation accuracies %v", null, nulls)
	}
	if null >= real {
		t.Errorf("null accuracy (%v) should be lower than the accuracy with the real labels (%v)", null, real)
	}
	if !slices.Equal(data.Labels, origLabels) {
		t.Errorf("NullRepeatedCrossValidation modified the labels of the original data")
	}
}

// With opts.Conc true, nothing must be sent through the channels (the function
// is not concurrent), and opts.Conc must be restored afterwards, both on success
// and on error.
func TestNullRepeatedCrossValidationRestoresConc(t *testing.T) {
	data := nullToyData(30, 2)
	newOpts := func(O *boo.Options) *Options {
		return &Options{O: O, Conc: true,
			Acc: make(chan float64, 1), Err: make(chan error, 1), Ochan: make(chan *boo.Options, 1)}
	}

	opts := newOpts(nullTestOptions())
	if _, _, err := NullRepeatedCrossValidation(data, 3, 2, 2, opts); err != nil {
		t.Fatalf("NullRepeatedCrossValidation returned an error: %v", err)
	}
	if !opts.Conc {
		t.Errorf("opts.Conc not restored after a successful call")
	}
	if len(opts.Acc) != 0 || len(opts.Err) != 0 || len(opts.Ochan) != 0 {
		t.Errorf("NullRepeatedCrossValidation sent values through the channels")
	}

	opts = newOpts(nil) // nil boo options make the cross-validation fail
	if _, _, err := NullRepeatedCrossValidation(data, 3, 2, 2, opts); err == nil {
		t.Fatal("expected an error when *Options.O is nil, got nil")
	}
	if !opts.Conc {
		t.Errorf("opts.Conc not restored after a failed call")
	}
}

func TestNullRepeatedCrossValidationNilOptions(t *testing.T) {
	data := mustLoadTrainSVM(t)
	_, _, err := NullRepeatedCrossValidation(data, 3, 2, 2, &Options{O: nil, Conc: false})
	if err == nil {
		t.Fatal("expected an error when *Options.O is nil, got nil")
	}
}

func TestNullRepeatedCrossValidationInvalidArgs(t *testing.T) {
	data := nullToyData(30, 3)
	cases := []struct {
		name         string
		nreps, nperm int
	}{
		{"zero nreps", 0, 2},
		{"negative nreps", -1, 2},
		{"zero nperm", 2, 0},
		{"negative nperm", 2, -1},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			defer func() {
				if r := recover(); r != nil {
					t.Errorf("panicked instead of returning an error: %v", r)
				}
			}()
			_, _, err := NullRepeatedCrossValidation(data, 3, c.nreps, c.nperm, &Options{O: nullTestOptions()})
			if err == nil {
				t.Errorf("expected an error for nreps=%d, nperm=%d, got nil", c.nreps, c.nperm)
			}
		})
	}
}
