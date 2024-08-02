package boo

import (
	"fmt"

	"github.com/rmera/boo/utils"
)

// Contain options to create a multi-class classification ensamble.
type Options struct {
	XGB            bool
	Rounds         int
	MaxDepth       int
	EarlyStop      int //roundw without increased fit before we stop trying.
	LearningRate   float64
	Lambda         float64
	MinChildWeight float64
	Gamma          float64
	SubSample      float64
	ColSubSample   float64
	BaseScore      float64
	MinSample      int //the minimum samples in each tree
	TreeMethod     string
	//	EarlyStopRounds      int //stop after n consecutive rounds of no improvement. Not implemented yet.
	Verbose bool
	Loss    utils.LossFunc
}

// Returns a pointer to an Options structure with the default values
// for an XGBoost multi-class classification ensamble.
func DefaultXOptions() *Options {
	O := new(Options)
	O.XGB = true
	O.Rounds = 20
	O.SubSample = 0.8
	O.ColSubSample = 0.8
	O.Lambda = 1.5
	O.MinChildWeight = 3
	O.Gamma = 0.2
	O.MaxDepth = 5
	O.LearningRate = 0.3
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.EarlyStop = 10
	O.Loss = &utils.SQErrLoss{}
	O.Verbose = false //just for clarity
	O.MinSample = 5
	return O
}

func (o *Options) Equal(O *Options) bool {
	if O.XGB != o.XGB {
		return false
	}
	if O.Rounds != o.Rounds {
		return false
	}
	if O.SubSample != o.SubSample {
		return false
	}
	if O.ColSubSample != o.ColSubSample {
		return false
	}
	if O.Lambda != o.Lambda {
		return false
	}
	if O.MinChildWeight != o.MinChildWeight {
		return false
	}
	if O.Gamma != o.Gamma {

		return false
	}
	if O.MaxDepth != o.MaxDepth {
		return false
	}
	if O.LearningRate != o.LearningRate {
		return false
	}
	if O.BaseScore != o.BaseScore {
		return false
	}
	if O.TreeMethod != "exact" {
		return false
	}
	if O.Loss != o.Loss {
		return false
	}
	if O.Verbose != o.Verbose {
		return false
	}
	if O.MinSample != o.MinSample {
		return false
	}
	return true
}

func (o *Options) Clone() *Options {
	O := new(Options)
	O.XGB = o.XGB
	O.Rounds = o.Rounds
	O.SubSample = o.SubSample
	O.ColSubSample = o.ColSubSample
	O.Lambda = o.Lambda
	O.MinChildWeight = o.MinChildWeight
	O.Gamma = o.Gamma
	O.MaxDepth = o.MaxDepth
	O.LearningRate = o.LearningRate
	O.BaseScore = o.BaseScore
	O.TreeMethod = "exact"
	O.Loss = o.Loss
	O.Verbose = o.Verbose
	O.MinSample = o.MinSample
	return O

}

// Returns a pointer to an Options structure with the default
// options a for regular gradient boosting multi-class classification
// ensamble.
func DefaultGOptions() *Options {
	O := new(Options)
	O.XGB = false
	O.Rounds = 10
	O.MaxDepth = 4
	O.LearningRate = 0.1
	O.MinChildWeight = 3
	O.Loss = &utils.MSELoss{}

	return O
}

func DefaultOptions() *Options {
	return DefaultXOptions()
}

// Returns a string representation of the options
func (O *Options) String() string {
	if O.XGB {
		return fmt.Sprintf("xgboost %d r/%d md/%.3f lr/%.3f ss/%.3f bs/%.3f gam/%.3f lam/%.3f mcw/%.3f css", O.Rounds, O.MaxDepth, O.LearningRate, O.SubSample, O.BaseScore, O.Gamma, O.Lambda, O.MinChildWeight, O.ColSubSample)
	} else {
		return fmt.Sprintf("gboost %d r/%d md/%.3f lr/%.3f mcw", O.Rounds, O.MaxDepth, O.LearningRate, O.MinChildWeight)

	}

}

func (o *Options) Check() error {
	n := fmt.Errorf
	if o.Rounds <= 0 {
		return n("Round! %v", o.Rounds)
	}
	if o.LearningRate <= 0 {
		return n("LearningRate:", o.LearningRate)
	}
	if o.SubSample <= 0 || o.ColSubSample <= 0 {
		return n("SubsSample or ColSubSample %v %v", o.SubSample, o.ColSubSample)
	}
	ssmax := 1.0
	if o.SubSample > ssmax || o.ColSubSample > ssmax {
		return n("SubSample or ColSubSample %v %v", o.SubSample, o.ColSubSample)
	}
	if o.Lambda < 0 {
		return n("Lambda %v", o.Lambda)
	}
	if o.MinChildWeight < 1 {
		return n("MinChildWeight %d", o.MinChildWeight)
	}
	if o.Gamma < 0 {
		return n("Gamma %v", o.Gamma)
	}
	if o.MaxDepth < 2 {
		return n("MaxDepth %v", o.MaxDepth)
	}
	if o.MinSample < 1 {
		return n("MinSample %v", o.MinSample)
	}
	return nil
}
