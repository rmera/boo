package cv

import (
	"bufio"
	"fmt"
	"log"
	"os"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

// Runs a nfold-cross-validation test of the options in opts the data D. It can return the results
// directly or sent them through channels.
func MultiClassCrossValidation(D *utils.DataBunch, nfold int, opts *Options) (float64, error) {
	var accus float64
	n, sampler, err := utils.CrossValidationSamples(D, nfold, true)
	if err != nil {
		if n == 0 {
			if opts.Conc {
				opts.Err <- err
				opts.Acc <- -1
				opts.Ochan <- nil
			}
			return 0, err
		}
		log.Printf("Only %d-fold can be performed due to sample size (%d). Error: %v", n, len(D.Data), err)
	}
	var i int
	for i = 0; i < n; i++ {
		var b *boo.MultiClass
		train, test := sampler()
		if opts.O == nil {
			b = boo.NewMultiClass(train)
		} else {
			b = boo.NewMultiClass(train, opts.O)
		}
		if b.Rounds() <= 0 {
			log.Printf("The %d-th fold didn't produce a boosting ensemble, will continue with the others", n)

			continue
		}
		a := b.Accuracy(test)
		accus += a
	}
	if opts.Conc {
		opts.Err <- nil
		opts.Acc <- accus / float64(i) //I use i and not n in case we skip some steps.
		opts.Ochan <- opts.O
	}
	return accus / float64(n), nil
}

// Contains options limiting the search-space
// of a cross-validation-based grid search for best hyperparameters.
// in all cases the 3 numbers are: initial, final, step
type GridOptions struct {
	XGB            bool
	EarlyStop      int
	Rounds         [3]int
	MaxDepth       [3]int
	LearningRate   [3]float64
	Gamma          [3]float64
	Lambda         [3]float64
	SubSample      [3]float64
	ColSubSample   [3]float64
	MinChildWeight [3]float64
	Step           float64
	DeltaFraction  float64
	Verbose        bool
	NSteps         int
	Central        bool
	NCPUs          int
	WriteBest      bool
	Regression     bool
}

func (o *GridOptions) Clone() *GridOptions {
	ret := new(GridOptions)
	ret.WriteBest = o.WriteBest
	ret.EarlyStop = o.EarlyStop
	ret.XGB = o.XGB
	ret.Rounds = o.Rounds
	ret.MaxDepth = o.MaxDepth
	ret.LearningRate = o.LearningRate
	ret.Gamma = o.Gamma
	ret.Lambda = o.Lambda
	ret.SubSample = o.SubSample
	ret.ColSubSample = o.ColSubSample
	ret.MinChildWeight = o.MinChildWeight
	ret.Step = o.Step
	ret.DeltaFraction = o.DeltaFraction
	ret.NSteps = o.NSteps
	ret.Central = o.Central
	ret.Verbose = o.Verbose
	ret.NCPUs = o.NCPUs
	ret.Regression = o.Regression
	return ret
}

// Default options for crossvalidation grid search for
// gradient boosting hyperparameters. Note that these are not
// necessarily good choices. These defaults are NOT considered part of the API
func DefaultGGridOptions() *GridOptions {
	ret := new(GridOptions)
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
	ret.EarlyStop = boo.DefaultGOptions().EarlyStop
	return ret
}

// Default options for crossvalidation grid search for
// XGBoost hyperparameters. Note that these are not necessaritly
// good choices. These defaults are NOT considered part of the API.
func DefaultXGridOptions() *GridOptions {
	ret := new(GridOptions)
	ret.XGB = true
	ret.Rounds = [3]int{20, 1000, 100}
	ret.MaxDepth = [3]int{3, 6, 1}
	ret.LearningRate = [3]float64{0.01, 0.5, 0.15}
	ret.Gamma = [3]float64{0.0, 0.5, 0.1}
	ret.Lambda = [3]float64{0.5, 2.0, 0.2}
	ret.SubSample = [3]float64{0.6, 0.9, 0.1}
	ret.ColSubSample = [3]float64{0.6, 0.9, 0.1}
	ret.MinChildWeight = [3]float64{3, 5, 1}
	ret.Step = 0.1
	ret.DeltaFraction = 0.05
	ret.NSteps = 6
	ret.Central = true
	ret.EarlyStop = boo.DefaultXOptions().EarlyStop
	ret.Verbose = false
	ret.NCPUs = 1
	return ret
}

func DefaultGridOptions() *GridOptions {
	return DefaultXGridOptions()
}

type Options struct {
	O     *boo.Options
	Conc  bool
	Acc   chan float64
	Ochan chan *boo.Options
	Err   chan error
}

// Runs a nfold-cross-validation-based grid search for best hyperparameters within the search space limited by options.
func Grid(data *utils.DataBunch, nfold int, options ...*GridOptions) (float64, []float64, *boo.Options, error) {
	var o *GridOptions
	if len(options) > 0 && options[0] != nil {
		o = options[0]
	} else {
		o = DefaultXGridOptions()
	}
	defaultoptions := boo.DefaultGOptions
	if o.XGB {
		defaultoptions = boo.DefaultXOptions
	}
	var finaloptions *boo.Options
	accuracies := make([]float64, 0, 100)
	bestacc := 0.0
	accs := make([]chan float64, o.NCPUs)
	errs := make([]chan error, o.NCPUs)

	os := make([]chan *boo.Options, o.NCPUs)
	for i := range o.NCPUs {
		accs[i] = make(chan float64)
		errs[i] = make(chan error)
		os[i] = make(chan *boo.Options)
	}
	//welcome to nested-hell. Sorry.
	cpus := 0
	for rounds := o.Rounds[0]; rounds <= o.Rounds[1]; rounds += o.Rounds[2] {
		for cw := o.MinChildWeight[0]; cw <= o.MinChildWeight[1]; cw += o.MinChildWeight[2] {
			if o.Verbose {
				fmt.Println("Rounds: ", rounds, "MinChildWeight:", cw)
			}
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
									t.Lambda = lam
									t.Gamma = gam
									t.SubSample = ss
									t.MinChildWeight = cw
									t.XGB = o.XGB
									t.EarlyStop = o.EarlyStop
									t.Regression = o.Regression
									conc := &Options{O: t, Acc: accs[cpus], Err: errs[cpus], Ochan: os[cpus], Conc: true}
									go MultiClassCrossValidation(data, nfold, conc)
									cpus++
									if cpus == o.NCPUs {
										var err error
										bestacc, finaloptions, err = rescueConcValues(errs, accs, os, bestacc, finaloptions, o.Verbose, o.WriteBest, data)

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

// Rescues cross-validation results from the given channels (errors, accuracy and the corresponding boosting options)
func rescueConcValues(errors []chan error, accs []chan float64, opts []chan *boo.Options, bestacc float64, bestop *boo.Options, verbose bool, writebest bool, data *utils.DataBunch) (float64, *boo.Options, error) {
	var err error
	var tmpacc float64
	var tmpop *boo.Options
	for i, v := range errors {
		err = <-v
		if err != nil {
			return -1, nil, err
		}
		tmpacc = <-accs[i]
		if tmpacc < 0 {
			return -1, nil, fmt.Errorf("grads zero") //not a real error, just that the optimizatio is over.
		}
		tmpop = <-opts[i]
		if tmpacc > bestacc {
			bestacc = tmpacc
			bestop = tmpop
			if verbose {
				if bestop.Regression {
					fmt.Printf("New Best RMSD: %.2f, %s\n", 1/bestacc, bestop.String())
				} else {
					fmt.Printf("New Best Accuracy %.0f%%, %s\n", bestacc, bestop.String())
				}
			}
			if writebest {
				_ = writeBest(data, bestacc, bestop)
			}
		}
	}
	return bestacc, bestop, nil

}

func writeBest(data *utils.DataBunch, bestacc float64, bestop *boo.Options) error {
	if bestop.Regression {
		bestacc = 1 / bestacc
	}
	name := fmt.Sprintf("xgbmodel%d.json", int(bestacc))
	f, err := os.Create(name)
	if err != nil {
		return err
	}
	boosted := boo.NewMultiClass(data, bestop)
	bf := bufio.NewWriter(f)
	err = boo.JSONMultiClass(boosted, "softmax", bf)
	if err != nil {
		return err
	}
	bf.Flush()
	f.Close()
	return nil
}
