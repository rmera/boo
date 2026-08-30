package an

/*
This code is a direct translation of the code by S. Nogeira avaliable at: https://github.com/nogueirs/JMLR2018
Even some of the comments are taken directly from there. The test compares the result of these functions to their
original Python code.
*/

import (
	"fmt"
	"math"
	"slices"

	"github.com/rmera/boo"
	"github.com/rmera/boo/strap"
	"github.com/rmera/boo/utils"
	"gonum.org/v1/gonum/floats"
)

func checkMat(f [][]float64) {
	if len(f) == 0 {
		panic("Given a nil [][]float64!")
	}
	if f[0] == nil {
		panic("The [][]float64 has to contain at least 1 non nil slice!")
	}

}

func col(f [][]float64, i int, dst ...[]float64) []float64 {
	var d []float64
	if len(dst) > 0 && len(dst[0]) == len(f) {
		d = dst[0]
	}
	d = make([]float64, len(f))
	for j, v := range f {
		d[j] = v[i]
	}
	return d
}

func Mean(f []float64) float64 {
	return floats.Sum(f) / float64(len(f))
}

// Dimension of the matrix given by dims.
// Will panic if f or f[0] is nil
func dims(f [][]float64) (int, int) {
	checkMat(f)
	return len(f), len(f[0])
}

// Returns a slice of floats with the mean of each column of Z
func colMeans(Z [][]float64, tmpmatrix ...[]float64) []float64 {
	var tmp []float64
	r, c := dims(Z)
	//we try to recycle the tmp form previous calls but we'll get a new one
	//if not given one.
	if len(tmpmatrix) > 0 && len(tmpmatrix[0]) == r {
		tmp = tmpmatrix[0]
	} else {
		tmp = make([]float64, r)
	}

	means := make([]float64, c) //one mean for each column

	for i, _ := range means {
		tmp = col(Z, i, tmp)
		means[i] = Mean(tmp)
	}
	return means

}

// Returns a slice of floats with the mean or sum (if sum is given and true) of each column of Z
func rowMeans(Z [][]float64, sum ...bool) []float64 {
	r, c := dims(Z)
	//we try to recycle the tmp form previous calls but we'll get a new one
	//if not given one.

	means := make([]float64, r) //one mean for each row

	den := float64(c)
	if len(sum) > 0 && sum[0] {
		den = 1.0
	}
	for i, _ := range means {
		row := Z[i]
		sum := floats.Sum(row)
		means[i] = sum / den
	}
	return means
}

// Z is row-major
func matrixThings(Z [][]float64) (float64, float64, []float64, float64) {
	iM := len(Z)
	id := len(Z[0])
	M := float64(iM)          //number of  feature sets
	d := float64(id)          //numer of features
	hatPF := colMeans(Z)      //frequency of selection of each feature
	kbar := floats.Sum(hatPF) //average number of selected features in the sets
	return M, d, hatPF, kbar

}

/*
From:  S. Nogueira, K. Sechidis, G. Brown, J. Mach. Learn. Res. 2017, 18, 174:1-174:54.

The comments are taken from the original Python functions that accompany the reference above.

Let us assume we have M>1 feature sets and d>0 features in total.
This function computes the stability estimate as given in Definition 4 in  [1].

INPUT: A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d).

	Each row of the binary matrix represents a feature set, where a 1 at the f^th position
	means the f^th feature has been selected and a 0 means it has not been selected.

OUTPUT: The stability of the feature selection procedure
*/
func FeatStability(Z [][]float64) float64 {
	r, _, _, _, _ := featStabilityRaw(Z)
	return r
}

func featStabilityRaw(Z [][]float64) (float64, float64, float64, []float64, float64) {
	M, d, hatPF, kbar := matrixThings(Z)
	denom := (kbar / d) * (1 - kbar/d)

	hatPFminus := make([]float64, len(hatPF))
	for i, v := range hatPF {
		hatPFminus[i] = v * (1.0 - v)
	}
	meanshat := floats.Sum(hatPFminus) / float64(len(hatPFminus))

	mult := ((M / (M - 1)) * meanshat) / denom

	return 1 - mult, M, d, hatPF, kbar
}

/*
From:  S. Nogueira, K. Sechidis, G. Brown, J. Mach. Learn. Res. 2017, 18, 174:1-174:54.

Let us assume we have M>1 feature sets and d>0 features in total.
This function computes the stability estimate and its variance as given in [1].

INPUT: A BINARY matrix Z (given as a list or as a numpy.ndarray of size M*d, raises a ValueError exception otherwise).

	Each row of the binary matrix represents a feature set, where a 1 at the f^th position
	means the f^th feature has been selected and a 0 means it has not been selected.

OUTPUT: A dictionnary where the key 'stability' provides the corresponding stability value #

	and where the key 'variance' provides the variance of the stability estimate

”
*/
func StabilityAndVariance(Z [][]float64) (float64, float64) {
	stab, M, d, hatPF, kbar := featStabilityRaw(Z)
	k := rowMeans(Z, true) //number of features selected in each of the M feature sets
	denom := (kbar / d) * (1 - kbar/d)
	phi := make([]float64, len(Z))

	row := make([]float64, len(Z[0]))
	for i, _ := range phi {
		copy(row, Z[i])
		floats.Mul(row, hatPF)
		term1 := (floats.Sum(row) / float64(len(row)))
		term2 := (k[i] * kbar) / (d * d)
		term3 := 2*term2 - k[i]/d - kbar/d + 1
		phi[i] = (term1 - term2 + (stab/2)*term3) / denom
	}
	phiav := Mean(phi)
	for i, v := range phi {
		phi[i] = math.Pow(v-phiav, 2)
	}
	vari := (4 / (M * M)) * floats.Sum(phi)
	return stab, vari
}

// Takes a set of int slices each representing a group of features. Say
// you get 2 groups, one with features 0, 2,3 and one with 1,5
// groups=[][]int{{0,2,3},{1,5}}
// retuns a function that takes a slice of ints and
// returns a slice of ints and an int. The int is always
// the number of groups in the original groups slice (2 in our example)
// The slice is a grouping of the input slice. Say  you get the slice []int{0,2}
// the output will be just []{0} since only the first (zero) groups is represented
// in the input slice.
func MakeGrouperFunc(groups [][]int) func([]int) ([]int, int) {
	nu := len(groups)
	f := func(l []int) ([]int, int) {
		ret := make([]int, len(groups))
		ret2 := make([]int, 0, len(groups))
		for _, v := range l {
			for i, w := range groups {
				if slices.Contains(w, v) {
					ret[i] = 1
					break
				}
			}
		}
		for i, v := range ret {
			if v == 1 {
				ret2 = append(ret2, i)
			}
		}
		return ret2, nu
	}
	return f
}

// Obtains the  Nogeira et al. feature stability, and it's variance, for models with options O trained on NBoot
// bootstrapped sets from those in D.  Features may be grouped for the analysis by the options groupfunc that
// returns a grouped feature set and the total number of groups from a 'raw' feature set.
// if given, the function groupfunc[0] must be guaranteed to return a slice the same length every time, even
// if you give it a 0 lenght slice
// If you use this function Please cite:
// S. Nogueira, K. Sechidis, G. Brown, J. Mach. Learn. Res. 2017, 18, 174:1-174:54.
func StabilityOnDataVar(D *utils.DataBunch, O *boo.Options, NBoot, Nfeat int, groupfunc ...func([]int) ([]int, int)) (float64, float64) {
	//This is the main API function, what I would expect end users to call the most.
	var group func([]int) ([]int, int) = func(f []int) ([]int, int) { return f, len(D.Keys) }
	if len(groupfunc) > 0 {
		group = groupfunc[0]
	}
	bD := D.Copy(true)
	if Nfeat <= 0 {
		Nfeat = len(D.Keys) //all the features
	}
	fvecs := make([][]float64, 0, len(D.Keys))
	for i := range NBoot {
		bD.Data = strap.Bootstrap(D.Data, bD.Data)
		//fmt.Println(bD.Data) ////////////////////////////////////////////////////////////
		xgb := boo.NewMultiClass(bD, O)
		feat, err := xgb.FeatureImportance()
		if err != nil {
			panic(fmt.Sprintf("Failed with bootstrapped xgb mode %d: %s", i, err.Error()))
		}
		features, lenrow := group(feat.Feats(Nfeat)) //note that we are grouping the Nfeat most important features.
		featrow := make([]float64, lenrow)
		for _, v := range features {
			featrow[v] = 1.0
		}

		fvecs = append(fvecs, featrow)
	}
	fmt.Println(fvecs) ///////////////////
	return StabilityAndVariance(fvecs)
}
