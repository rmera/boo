package xgb

import (
	"fmt"
	"log"

	"github.com/rmera/learn"
)

func ConcMultiClassCrossValidation(D *learn.DataBunch, nfold int, opts *COptions) {
	var accus float64
	n, sampler, err := learn.CrossValidationSamples(D, nfold, true)
	if err != nil {
		if n == 0 {
			opts.err <- err
			opts.acc <- -1
			opts.ochan <- nil
			return
		}
		log.Printf("Only %d-fold can be performed due to sample size (%d). Error: %v", n, len(D.Data), err)
	}
	for i := 0; i < n; i++ {
		var b *MultiClass
		train, test := sampler()
		if opts.o == nil {
			b = NewMultiClass(train)
		} else {
			b = NewMultiClass(train, opts.o)
		}
		a := b.Accuracy(test)
		accus += a
	}
	opts.err <- nil
	opts.acc <- accus / float64(n)
	opts.ochan <- opts.o
	return

}

// does
func MultiClassCrossValidation(D *learn.DataBunch, nfold int, opts ...*Options) (float64, error) {
	var accus float64
	n, sampler, err := learn.CrossValidationSamples(D, nfold)
	if err != nil {
		if n == 0 {
			return -1, err
		}
		log.Printf("Only %d-fold can be performed due to sample size (%d). Error: %v", n, len(D.Data), err)
	}
	for i := 0; i < n; i++ {
		train, test := sampler()
		b := NewMultiClass(train)
		a := b.Accuracy(test)
		//		fmt.Println("Accuracy in train:", b.Accuracy(train), ". Accuracy in test:", a) //////////////////
		accus += a
		//	fmt.Println(i, "-fold accuracy:", a) /////////////////////////////
	}
	return accus / float64(n), nil

}

// in all cases the 3 numbers are: initial, final, step
type CVGridOptions struct {
	Rounds         [3]int
	MaxDepth       [3]int
	LearningRate   [3]float64
	Gamma          [3]float64
	Lambda         [3]float64
	SubSample      [3]float64
	MinChildWeight [3]float64
	NCPUs          int
}

func DefaultCVGridOptions() *CVGridOptions {
	ret := new(CVGridOptions)
	ret.Rounds = [3]int{20, 1000, 100}
	ret.MaxDepth = [3]int{3, 6, 1}
	ret.LearningRate = [3]float64{0.01, 0.5, 0.15}
	ret.Gamma = [3]float64{0.0, 0.5, 0.1}
	ret.Lambda = [3]float64{0.5, 2.0, 0.2}
	ret.SubSample = [3]float64{0.6, 0.9, 0.1}
	ret.MinChildWeight = [3]float64{3, 5, 1}
	ret.NCPUs = 1
	return ret
}

type COptions struct {
	o     *Options
	acc   chan float64
	ochan chan *Options
	err   chan error
}

func ConcCVGrid(data *learn.DataBunch, nfold int, verbose bool, options ...*CVGridOptions) (float64, []float64, *Options, error) {
	var o *CVGridOptions
	if len(options) > 0 && options[0] != nil {
		o = options[0]
	} else {
		o = DefaultCVGridOptions()
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
								conc := &COptions{o: t, acc: accs[cpus], err: errs[cpus], ochan: os[cpus]}
								go ConcMultiClassCrossValidation(data, nfold, conc)
								cpus++
								if cpus == o.NCPUs {
									var err error
									bestacc, finaloptions, err = rescueConcValues(errs, accs, os, bestacc, finaloptions, verbose)

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

func CVGrid(data *learn.DataBunch, nfold int, verbose bool, options ...*CVGridOptions) (float64, []float64, *Options, error) {
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
