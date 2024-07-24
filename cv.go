package learn

import (
	"fmt"
	"log"

	"github.com/rmera/chemlearn/utils"
)

func MultiClassCrossValidation(D *utils.DataBunch, nfold int, xgboost bool, opts *CVOptions) (float64, error) {
	var accus float64
	n, sampler, err := utils.CrossValidationSamples(D, nfold, true)
	if err != nil {
		if n == 0 {
			if opts.Conc {
				opts.err <- err
				opts.acc <- -1
				opts.ochan <- nil
			}
			return 0, err
		}
		log.Printf("Only %d-fold can be performed due to sample size (%d). Error: %v", n, len(D.Data), err)
	}
	for i := 0; i < n; i++ {
		var b *MultiClass
		train, test := sampler()
		if opts.O == nil {
			b = NewMultiClass(train, xgboost)
		} else {
			b = NewMultiClass(train, xgboost, opts.O)
		}
		a := b.Accuracy(test)
		accus += a
	}
	if opts.Conc {
		opts.err <- nil
		opts.acc <- accus / float64(n)
		opts.ochan <- opts.O
	}
	return accus / float64(n), nil

}

// in all cases the 3 numbers are: initial, final, step
type CVGridOptions struct {
	XGB            bool
	Rounds         [3]int
	MaxDepth       [3]int
	LearningRate   [3]float64
	Gamma          [3]float64
	Lambda         [3]float64
	SubSample      [3]float64
	ColSubSample   [3]float64
	MinChildWeight [3]float64
	Verbose        bool
	NCPUs          int
}

// Default options for crossvalidation grid search for
// gradient boosting hyperparameters. Note that these are not
// necessarily good choices.
func DefaultGCVGridOptions() *CVGridOptions {
	ret := new(CVGridOptions)
	ret.XGB = false
	ret.Rounds = [3]int{2, 100, 1}
	ret.MaxDepth = [3]int{2, 10, 1}
	ret.LearningRate = [3]float64{0.01, 0.8, 0.1}
	ret.MinChildWeight = [3]float64{2, 6, 1}
	//The point of this is to ensure these are not looped over,
	//only one "iteration" is taken on each, as the "upper limit" is
	//smaller than the lower limit+step.
	ret.Gamma = [3]float64{0.0, 1, 2}
	ret.Lambda = [3]float64{0, 1, 2}
	ret.SubSample = [3]float64{1, 1, 2}
	ret.ColSubSample = [3]float64{1, 1, 2}
	ret.Verbose = false
	return ret
}

// Default options for crossvalidation grid search for
// XGBoost hyperparameters. Note that these are not necessaritly
// good choices.
func DefaultXCVGridOptions() *CVGridOptions {
	ret := new(CVGridOptions)
	ret.XGB = true
	ret.Rounds = [3]int{20, 1000, 100}
	ret.MaxDepth = [3]int{3, 6, 1}
	ret.LearningRate = [3]float64{0.01, 0.5, 0.15}
	ret.Gamma = [3]float64{0.0, 0.5, 0.1}
	ret.Lambda = [3]float64{0.5, 2.0, 0.2}
	ret.SubSample = [3]float64{0.6, 0.9, 0.1}
	ret.ColSubSample = [3]float64{0.6, 0.9, 0.1}
	ret.MinChildWeight = [3]float64{3, 5, 1}
	ret.Verbose = false
	ret.NCPUs = 1
	return ret
}

type CVOptions struct {
	O     *Options
	Conc  bool
	acc   chan float64
	ochan chan *Options
	err   chan error
}

func ConcCVGrid(data *utils.DataBunch, nfold int, xgboost bool, options ...*CVGridOptions) (float64, []float64, *Options, error) {
	var o *CVGridOptions
	if len(options) > 0 && options[0] != nil {
		o = options[0]
	} else {
		if xgboost {
			o = DefaultXCVGridOptions()

		} else {
			o = DefaultGCVGridOptions()
		}
	}
	o.XGB = xgboost //just in case
	defaultoptions := DefaultGOptions
	if xgboost {
		defaultoptions = DefaultXOptions
	}
	var finaloptions *Options
	accuracies := make([]float64, 0, 100)
	bestacc := 0.0
	accs := make([]chan float64, o.NCPUs)
	errs := make([]chan error, o.NCPUs)

	os := make([]chan *Options, o.NCPUs)
	for i := range o.NCPUs {
		accs[i] = make(chan float64)
		errs[i] = make(chan error)
		os[i] = make(chan *Options)
	}
	//welcome to nested-hell. Sorry.

	cpus := 0
	for cw := o.MinChildWeight[0]; cw <= o.MinChildWeight[1]; cw += o.MinChildWeight[2] {
		for rounds := o.Rounds[0]; rounds <= o.Rounds[1]; rounds += o.Rounds[2] {
			for md := o.MaxDepth[0]; md <= o.MaxDepth[1]; md += o.MaxDepth[2] {
				for css := o.ColSubSample[0]; css <= o.ColSubSample[1]; css += o.ColSubSample[2] {
					for lr := o.LearningRate[0]; lr <= o.LearningRate[1]; lr += o.LearningRate[2] {
						for lam := o.Lambda[0]; lam <= o.Lambda[1]; lam += o.Lambda[2] {
							for gam := o.Gamma[0]; gam <= o.Gamma[1]; gam += o.Gamma[2] {
								for ss := o.SubSample[0]; ss <= o.SubSample[1]; ss += o.SubSample[2] {

									t := defaultoptions()
									t.ColSubSample = css
									t.LearningRate = lr
									t.MaxDepth = md
									t.Rounds = rounds
									t.RegLambda = lam
									t.Gamma = gam
									t.SubSample = ss
									t.MinChildWeight = cw
									conc := &CVOptions{O: t, acc: accs[cpus], err: errs[cpus], ochan: os[cpus], Conc: true}
									go MultiClassCrossValidation(data, nfold, xgboost, conc)
									cpus++
									if cpus == o.NCPUs {
										var err error
										bestacc, finaloptions, err = rescueConcValues(errs, accs, os, bestacc, finaloptions, o.Verbose)

										if err != nil {
											return -1, nil, nil, err
										}
										cpus = 0
									}

								}

							}
						}
					}
				}
			}
		}
	}
	return bestacc, accuracies, finaloptions, nil
}

func rescueConcValues(errors []chan error, accs []chan float64, opts []chan *Options, bestacc float64, bestop *Options, verbose bool) (float64, *Options, error) {
	var err error
	var tmpacc float64
	var tmpop *Options
	for i, v := range errors {
		err = <-v
		if err != nil {
			return -1, nil, err
		}
		tmpacc = <-accs[i]
		tmpop = <-opts[i]
		if tmpacc > bestacc {
			bestacc = tmpacc
			bestop = tmpop
			if verbose {
				fmt.Printf("New Best Accuracy %.0f%%, %s\n", bestacc, bestop.String())
			}
		}

	}
	return bestacc, bestop, nil

}

/*
func CVGrid(data *utils.DataBunch, nfold int, verbose bool, options ...*CVGridOptions) (float64, []float64, *Options, error) {
	var o *CVGridOptions
	if len(options) > 0 && options[0] != nil {
		o = options[0]
	} else {
		o = DefaultCVGridOptions()
	}
	var finaloptions *Options
	accuracies := make([]float64, 0, 100)
	bestacc := 0.0
	//welcome to nested-hell. Sorry.
	for cw := o.MinChildWeight[0]; cw <= o.MinChildWeight[1]; cw += o.MinChildWeight[2] {
		for rounds := o.Rounds[0]; rounds <= o.Rounds[1]; rounds += o.Rounds[2] {
			for md := o.MaxDepth[0]; md <= o.MaxDepth[1]; md += o.MaxDepth[2] {
				for lr := o.LearningRate[0]; lr <= o.LearningRate[1]; lr += o.LearningRate[2] {
					for lam := o.Lambda[0]; lam <= o.Lambda[1]; lam += o.Lambda[2] {
						for gam := o.Gamma[0]; gam <= o.Gamma[1]; gam += o.Gamma[2] {
							for ss := o.SubSample[0]; ss <= o.SubSample[1]; ss += o.SubSample[2] {

								t := DefaultOptions()
								t.LearningRate = lr
								t.MaxDepth = md
								t.Rounds = rounds
								t.RegLambda = lam
								t.Gamma = gam
								t.SubSample = ss
								t.MinChildWeight = cw
								acc, err := MultiClassCrossValidation(data, nfold, t)
								if err != nil {
									return 0, nil, nil, err
								}
								accuracies = append(accuracies, acc)
								if acc > bestacc {
									if verbose {
										fmt.Printf("New Best Accuracy %.0f%%, %s\n", acc, t.String())
									}
									finaloptions = t
									bestacc = acc
								}

							}
						}

					}
				}
			}
		}
	}
	return bestacc, accuracies, finaloptions, nil
}

*/
