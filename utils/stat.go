package utils

import (
	"fmt"
	"math/rand/v2"
	"slices"
)

func isInPrevious(i int, sam [][]int) bool {
	if len(sam) == 0 {
		return false
	}
	for _, v := range sam {
		if slices.Contains(v, i) {
			return true
		}
	}
	return false
}

// Fills ori with the data from the vectors of ori with indexes in toadd. It also fills the labels
// witht the labels corresponding to those vectors in ori (if the labels exist) and fills the keys
// with all the keys in ori, if they exist. If docopy is true, then the new data and keys in dest
// will be copies, not references, of/to those in ori. (labels are always copies). The filled dest
// is returned.
func fillDataBunch(ori, dest *DataBunch, toadd []int, docopy bool) *DataBunch {
	dest.Data = make([][]float64, 0, len(toadd))
	dest.Labels = make([]int, 0, len(toadd))

	for _, v := range toadd {
		var add []float64
		if docopy {
			add = make([]float64, len(ori.Data[v]))
			copy(add, ori.Data[v])
		} else {
			add = ori.Data[v]
		}
		dest.Data = append(dest.Data, add)
		if len(ori.Labels) > v {
			dest.Labels = append(dest.Labels, ori.Labels[v])
		}
		if len(ori.FloatLabels) > v {
			dest.FloatLabels = append(dest.FloatLabels, ori.FloatLabels[v])
		}

	}
	if len(dest.Keys) > 0 {
		if docopy {
			dest.Keys = make([]string, len(ori.Keys))
			copy(dest.Keys, ori.Keys)
		} else {
			dest.Keys = ori.Keys
		}
	}
	return dest

}

// Returns an int "n", and a function that the first n times is called, will return a "folded" training set
// and test set. After the function has been called n times, it will return nil, nil. If the fold requested
// is too large for the dataset, an error will be returned and a smaller fold will be used. If not nfold is
// requested, 5-fold will be used.
func CrossValidationSamples(data *DataBunch, nfold int, usecopies ...bool) (int, func() (*DataBunch, *DataBunch), error) {
	nf := 5
	var docopy bool
	if len(usecopies) > 0 {
		docopy = usecopies[0]
	}
	const minleftout int = 2
	const minfold int = 2
	var err error
	if nfold > 0 {
		nf = nfold
	}
	tot := len(data.Data)
	tried := make([]int, 0, 5)
	for ; tot/nf < minleftout; nf-- {
		err = fmt.Errorf("The %d-fold crossvalidation requested leaves only %d sample(s) per fold, when the minimum allowed is %d. Will use %d-fold.N-folds already tried: %v", nf, tot/nf, minleftout, nf-1, tried)
		tried = append(tried, nf)
	}
	if nf < minfold {
		return 0, nil, fmt.Errorf("Too few samples for crossvalidation. The final n-fold %d is smaller than the minimum %d. Folds tried: %v", nf, minfold, tried)
	}

	nsamples := tot / nf
	folds := make([][]int, nf)
	//Note that this scheme means that some samples might be left out of the process altogether.
	for i := range folds {
		for j := 0; j < nsamples; j++ {
			var n int
			for n = rand.IntN(tot); isInPrevious(n, folds); n = rand.IntN(tot) {
			}
			folds[i] = append(folds[i], n)
		}

	}
	folds = fillRemaining(folds, tot) //since tot/nf might leave some samples behind,  those samples, if any, are
	//rescued here and added to the last fold.

	calls := 0
	ret := func() (*DataBunch, *DataBunch) {
		if calls >= nf {
			return nil, nil
		}
		testindexes := folds[calls]
		test := &DataBunch{}
		test = fillDataBunch(data, test, testindexes, docopy)

		train := &DataBunch{}
		trainindexes := make([]int, 0, nsamples*nf-1)
		for i, v := range folds {
			if i == calls {
				continue
			}
			trainindexes = append(trainindexes, v...)
		}
		train = fillDataBunch(data, train, trainindexes, docopy)
		//	fmt.Println("test,train", testindexes, trainindexes) //////////////////
		calls++
		return train, test
	}
	return nf, ret, err
}

// given a set of tot samples, some of which might not be present in any of the slices in folds,
// add those not present to the last fold.
func fillRemaining(folds [][]int, tot int) [][]int {
	last := len(folds) - 1
	for i := 0; i < tot; i++ {
		if !isInPrevious(i, folds) {
			folds[last] = append(folds[last], i)
		}
	}
	return folds

}
