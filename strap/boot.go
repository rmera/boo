package strap

import (
	"fmt"
	"math/rand"
	"runtime"
	"slices"

	"github.com/rmera/boo/utils"
)

type RealNumber interface {
	~float32 | ~float64 | ~int | ~int8 | ~int16 | ~int32 | ~int64 | ~uint | ~uint8 | ~uint16 | ~uint32 | ~uint64
}

func BootstrapDataBunch(D *utils.DataBunch, dataholder [][]float64) *utils.DataBunch {
	var bdata [][]float64
	if len(dataholder) > 0 {
		bdata = dataholder
	} else {
		bdata = make([][]float64, len(D.Data))
	}
	bD := D.Copy(true)
	bD.Data = bdata
	bD.Data = Bootstrap(D.Data, bD.Data)
	return bD
}

// Bootstrap takes a slice of data and bootstraps a new slice
// by sampling data randomly with replacement.
// The ideas is that this function works easyly both concurrently and serially.
// If you want to use it concurrently, you just pass a channel of bools so that
// it can signal when it's done. Note that it doesn't copy elements (it can't,
// being generic) so, for a slice of pointers, you can get several elements that
// point to the same memory. If you want to have copies, make a new slice and copy
// the elements of what Bootstrap returns.
func Bootstrap[a any](data []a, place []a, ready ...chan bool) []a {
	var ret []a
	if place == nil {
		place = make([]a, len(data))
	}
	ret = place
	for i := range place {
		place[i] = data[rand.Intn(len(data))]
	}
	if ready != nil {
		ready[0] <- true
	}
	return ret //if you give a slice for "place", you can ignore this return value.
}

// Applies the function 'f' to 'nboot' bootstrapped samples from 'data', each of 'samples' size
// (commonly, 'sampled' is the same as len(data)) using npcus concurrent gorutines.
func FuncBootStrap[a any](data []a, f func([]a, int), samples, nboot, ncpus int) {
	scratch := make([][]a, ncpus)
	ready := make([]chan bool, ncpus)
	for i, _ := range scratch {
		scratch[i] = make([]a, samples)
		ready[i] = make(chan bool)
	}
	rounds := nboot / ncpus
	if nboot%ncpus != 0 {
		rounds++
	}
	for i := 0; i < rounds; i++ {
		limit := ncpus
		for j := 0; j < ncpus; j++ {
			id := (i * ncpus) + j //total bootstrap number
			if id >= nboot {
				limit = j
				break
			}
			go func(j int) {
				Bootstrap(data, scratch[j])
				f(scratch[j], id)
				ready[j] <- true

			}(j)
		}
		//we prevent the scratch to be re-used before the functions have finished
		for k := 0; k < limit; k++ {
			<-ready[k]

		}
	}
}

// Given the frequencies histogram data (where each elemetn is a bin), it uses nboot-bootstrap to return the
// upper and lower conflevel percent confidence intervals, corrected by the function correction.
func HistogramConfidence[N RealNumber](data []N, conflevel float64, nboot int, correction func(float64) float64) ([]float64, []float64) {
	ns := len(data)
	histos := make([][]float64, ns)
	for i, _ := range histos {
		histos[i] = make([]float64, nboot)
	}

	f := func(d []N, btnum int) {
		if len(d) != ns {
			panic(fmt.Sprintf("One of the bootstrapped histograms has an incorrect size: %d, should be %d", len(d), ns))
		}
		var tot N
		for _, v := range d {
			tot += v
		}
		ftot := float64(tot)
		for i, v := range d {
			histos[i][btnum] = float64(v) / ftot

		}
	}
	FuncBootStrap(data, f, ns, nboot, runtime.NumCPU()/2)
	confup := make([]float64, ns)
	confdown := make([]float64, ns)

	for i, v := range histos {
		slices.Sort(v)
		alpha := (1 - conflevel/100.0) / 2
		cu := 1 - alpha
		cd := alpha
		confup[i] = correction(v[int(cu*float64(len(v)))-1])
		confdown[i] = correction(v[int(cd*float64(len(v)))-1])
	}

	return confup, confdown

}
