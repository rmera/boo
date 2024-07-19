package gb

import (
	"fmt"
	"log"
	"sort"

	"github.com/rmera/learn"
)

func MultiClassCrossValidation(D *learn.DataBunch, nfold int, opts ...*Options) (float64, error) {
	var accus float64
	n, sampler, err := learn.CrossValidationSamples(D, nfold)
	if err != nil {
		if n == 0 {
			return -1, err
		}
		log.Printf("Only %d-fold can be performed due to sample size (%d),Error: %v", n, len(D.Data), err)
	}
	for i := 0; i < n; i++ {
		train, test := sampler()
		b := NewMultiClass(train, opts...)
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
	MinChildWeight [3]int
	Verbose        bool
}

func DefaultCVGridOptions() *CVGridOptions {
	ret := new(CVGridOptions)
	ret.Rounds = [3]int{2, 100, 1}
	ret.MaxDepth = [3]int{2, 10, 1}
	ret.LearningRate = [3]float64{0.01, 0.8, 0.1}
	ret.MinChildWeight = [3]int{2, 6, 1}
	ret.Verbose = false
	return ret
}

func CVGrid(data *learn.DataBunch, nfold int, options ...*CVGridOptions) (float64, []float64, *Options, error) {
	var o *CVGridOptions
	if len(options) > 0 && options[0] != nil {
		o = options[0]
	} else {
		o = DefaultCVGridOptions()
	}
	var finaloptions *Options
	accuracies := make([]float64, 0, 100)
	//oh boy
	bestacc := 0.0
	for mcw := o.MinChildWeight[0]; mcw <= o.MinChildWeight[1]; mcw += o.MinChildWeight[2] {
		for rounds := o.Rounds[0]; rounds <= o.Rounds[1]; rounds += o.Rounds[2] {
			for md := o.MaxDepth[0]; md <= o.MaxDepth[1]; md += o.MaxDepth[2] {
				for lr := o.LearningRate[0]; lr <= o.LearningRate[1]; lr += o.LearningRate[2] {
					t := DefaultOptions()
					t.LearningRate = lr
					t.MaxDepth = md
					t.Rounds = rounds
					t.MinChildWeight = mcw
					acc, err := MultiClassCrossValidation(data, nfold, t)
					if err != nil {
						return 0, nil, nil, err
					}
					accuracies = append(accuracies, acc)
					if acc > bestacc {
						finaloptions = t
						bestacc = acc
						if o.Verbose {
							fmt.Printf("New best accuracy %.3f, %v\n", acc, t)
						}
					}
				}
			}
		}

	}
	return bestacc, accuracies, finaloptions, nil

}

type optionset struct {
	o []*Options
	a []float64
}

func (o *optionset) Len() int           { return len(o.a) }
func (o *optionset) Less(i, j int) bool { return o.a[i] > o.a[j] } //descending order
func (o *optionset) Swap(i, j int) {
	o.a[i], o.a[j] = o.a[j], o.a[i]
	o.o[i], o.o[j] = o.o[j], o.o[i]
}

// Automatic refinement of a starting grid.
// tries several things, returns a list with the best accuracies and a list with the best options,
// both sorted descending order of accuracy
func CVGridofGrids(data *learn.DataBunch, nfold, maxdepth int, options ...*CVGridOptions) ([]float64, []*Options, error) {
	if maxdepth <= 0 {
		maxdepth = 8
	}
	var o *CVGridOptions
	if len(options) > 0 && options[0] != nil {
		o = options[0]
	} else {
		o = DefaultCVGridOptions()
		o.Rounds = [3]int{10, 15, 1}
		o.MaxDepth = [3]int{2, 4, 1}
		o.LearningRate = [3]float64{0.01, 0.51, 0.05}
	}
	var acc float64
	var best *Options
	opts := &optionset{o: make([]*Options, 0, 10), a: make([]float64, 0, 10)}
	var err error
	turns := -1
	lrtrials := 0
	for {
		turns++
		var keep bool
		acc, _, best, err = CVGrid(data, nfold, o)
		opts.a = append(opts.a, acc)
		opts.o = append(opts.o, best)
		fmt.Println(acc, best.String()) //////////////////////

		if err != nil {
			return nil, nil, err
		}
		if best.Rounds <= o.Rounds[0]+o.Rounds[2] {
			o.Rounds = [3]int{best.Rounds - 3*o.Rounds[2], best.Rounds, o.Rounds[2]}
			keep = true
		}
		if best.Rounds >= o.Rounds[1]-o.Rounds[2] {
			o.Rounds = [3]int{best.Rounds, best.Rounds + 3*o.Rounds[2], o.Rounds[2]}
			keep = true
		}

		if best.MaxDepth <= o.MaxDepth[0]+o.MaxDepth[2] {
			o.MaxDepth = [3]int{best.MaxDepth - 3*o.MaxDepth[2], best.MaxDepth, o.MaxDepth[2]}
			keep = true
		}
		if best.MaxDepth >= o.MaxDepth[1]-o.MaxDepth[2] && best.MaxDepth < maxdepth {
			o.MaxDepth = [3]int{best.MaxDepth, best.MaxDepth + 3*o.MaxDepth[2], o.MaxDepth[2]}
			keep = true
		}
		if !keep && lrtrials < 2 {
			keep = true
			newstep := o.LearningRate[2] / 5.0
			o.LearningRate = [3]float64{best.LearningRate - newstep*3, best.LearningRate + newstep*3, newstep}
		}
		if !keep {
			break
		}
	}
	sort.Sort(opts)
	return opts.a, opts.o, nil
}
