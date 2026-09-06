package boo

import (
	"strings"
	"testing"
)

func validOptions() *Options {
	O := DefaultXOptions()
	O.Rounds = 10
	O.LearningRate = 0.3
	O.SubSample = 0.8
	O.ColSubSample = 0.8
	O.Lambda = 1.0
	O.MinChildWeight = 1
	O.Gamma = 0
	O.MaxDepth = 3
	O.MinSample = 1
	return O
}

func TestOptionsCheck(t *testing.T) {
	if err := validOptions().Check(); err != nil {
		t.Fatalf("valid options rejected: %v", err)
	}
	cases := []struct {
		name string
		mod  func(*Options)
	}{
		{"Rounds<=0", func(o *Options) { o.Rounds = 0 }},
		{"LearningRate<=0", func(o *Options) { o.LearningRate = 0 }},
		{"SubSample<=0", func(o *Options) { o.SubSample = 0 }},
		{"ColSubSample<=0", func(o *Options) { o.ColSubSample = 0 }},
		{"SubSample>1", func(o *Options) { o.SubSample = 1.1 }},
		{"ColSubSample>1", func(o *Options) { o.ColSubSample = 1.1 }},
		{"Lambda<0", func(o *Options) { o.Lambda = -1 }},
		{"MinChildWeight<1", func(o *Options) { o.MinChildWeight = 0.5 }},
		{"Gamma<0", func(o *Options) { o.Gamma = -0.1 }},
		{"MaxDepth<2", func(o *Options) { o.MaxDepth = 1 }},
		{"MinSample<1", func(o *Options) { o.MinSample = 0 }},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			o := validOptions()
			c.mod(o)
			if err := o.Check(); err == nil {
				t.Errorf("Check() returned nil error, want an error for invalid %s", c.name)
			}
		})
	}
}

func TestOptionsCloneIsIndependent(t *testing.T) {
	o := DefaultXOptions()
	c := o.Clone()
	if !o.Equal(c) {
		t.Fatalf("freshly cloned options are not Equal to the original")
	}
	c.Rounds = o.Rounds + 1
	c.MaxDepth = o.MaxDepth + 1
	if o.Rounds == c.Rounds {
		t.Errorf("mutating the clone's Rounds also changed the original: they alias the same Options")
	}
	if o.MaxDepth == c.MaxDepth {
		t.Errorf("mutating the clone's MaxDepth also changed the original: they alias the same Options")
	}
}

func TestOptionsEqualDetectsDifferences(t *testing.T) {
	a := DefaultXOptions()
	fields := []struct {
		name string
		mod  func(*Options)
	}{
		{"XGB", func(o *Options) { o.XGB = !o.XGB }},
		{"Rounds", func(o *Options) { o.Rounds++ }},
		{"SubSample", func(o *Options) { o.SubSample += 0.01 }},
		{"ColSubSample", func(o *Options) { o.ColSubSample += 0.01 }},
		{"Lambda", func(o *Options) { o.Lambda += 0.01 }},
		{"MinChildWeight", func(o *Options) { o.MinChildWeight += 0.01 }},
		{"Gamma", func(o *Options) { o.Gamma += 0.01 }},
		{"MaxDepth", func(o *Options) { o.MaxDepth++ }},
		{"LearningRate", func(o *Options) { o.LearningRate += 0.01 }},
		{"BaseScore", func(o *Options) { o.BaseScore += 0.01 }},
		{"TreeMethod", func(o *Options) { o.TreeMethod = "hist" }},
		{"Verbose", func(o *Options) { o.Verbose = !o.Verbose }},
		{"MinSample", func(o *Options) { o.MinSample++ }},
		{"regression", func(o *Options) { o.Regression(true) }},
	}
	for _, f := range fields {
		t.Run(f.name, func(t *testing.T) {
			b := a.Clone()
			f.mod(b)
			if a.Equal(b) {
				t.Errorf("Equal() returned true after changing %s", f.name)
			}
		})
	}
}

func TestOptionsStringReflectsBoostingType(t *testing.T) {
	x := DefaultXOptions()
	if got := x.String(); !strings.Contains(got, "xgboost") {
		t.Errorf("String() for XGB options = %q, want it to contain %q", got, "xgboost")
	}
	g := DefaultGOptions()
	if got := g.String(); !strings.Contains(got, "gboost") {
		t.Errorf("String() for non-XGB options = %q, want it to contain %q", got, "gboost")
	}
}

func TestRegressionSetsIdentityActivation(t *testing.T) {
	o := DefaultXOptions()
	o.Regression(true)
	if got, want := o.Activation.Name(), "identity"; got != want {
		t.Errorf("Regression(true) left Activation.Name() = %q, want %q", got, want)
	}
}
