package learn

import (
	"fmt"
	"slices"

	"github.com/rmera/chemlearn/utils"
)

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

func setOptionsToMid(o *Options, co *CVGridOptions) {
	av := func(i1, i2 float64) float64 { return (i1 + i2) / 2 }
	avint := func(i1, i2 int) int { return (i1 + i2) / 2 }
	o.Rounds = avint(co.Rounds[0], co.Rounds[1])
	o.LearningRate = av(co.LearningRate[0], co.LearningRate[1])
	o.SubSample = av(co.SubSample[0], co.SubSample[1])
	o.ColSubSample = av(co.ColSubSample[0], co.ColSubSample[1])
	o.Lambda = av(co.Lambda[0], co.Lambda[1])
	o.MinChildWeight = av(co.MinChildWeight[0], co.MinChildWeight[1])
	o.Gamma = av(co.Gamma[0], co.Gamma[1])
	o.MaxDepth = avint(co.MaxDepth[0], co.MaxDepth[1])
}

func checkAgainstOptions(o *Options, co *CVGridOptions) error {
	n := fmt.Errorf
	if o.Rounds < co.Rounds[0] || o.Rounds > co.Rounds[1] {
		return n("Round! %v", o.Rounds)
	}
	if o.LearningRate < co.LearningRate[0] || o.LearningRate > co.LearningRate[1] {
		return n("LearningRate: %v", o.LearningRate)
	}
	if o.SubSample < co.SubSample[0] || o.ColSubSample < co.ColSubSample[0] {
		return n("SubsSample or ColSubSample %v %v", o.SubSample, o.ColSubSample)
	}
	if o.SubSample > co.SubSample[1] || o.ColSubSample > co.ColSubSample[1] {
		return n("SubsSample or ColSubSample %v %v", o.SubSample, o.ColSubSample)
	}
	if o.Lambda < co.Lambda[0] || o.Lambda > co.Lambda[1] {
		return n("Lambda %v", o.Lambda)
	}
	if o.MinChildWeight < co.MinChildWeight[0] || o.MinChildWeight > co.MinChildWeight[1] {
		return n("MinChildWeight %d", o.MinChildWeight)
	}
	if o.Gamma < co.Gamma[0] || o.Gamma > co.Gamma[1] {
		return n("Gamma %v", o.Gamma)
	}
	if o.MaxDepth < co.MaxDepth[0] || o.MaxDepth > co.MaxDepth[1] {
		return n("MaxDepth %v", o.MaxDepth)
	}
	if o.MinSample < 1 {
		return n("MinSample %v", o.MinSample)
	}
	return nil
}

func GradStep(Ori *Options, CVO *CVGridOptions, D *utils.DataBunch, step, fractiondelta float64, nfold int, central bool) *Options {
	O := Ori.Clone()
	fd := fractiondelta
	nt := (6 * 2) + 1
	last := nt - 1 //index of the last element
	accs := make([]chan float64, nt)
	errs := make([]chan error, nt)
	os := make([]chan *Options, nt)
	for i := 0; i < nt; i++ {
		accs[i] = make(chan float64)
		errs[i] = make(chan error)
		os[i] = make(chan *Options)
	}
	if !central {
		conc := &CVOptions{O: O, Acc: accs[last], Err: errs[last], Ochan: os[last], Conc: true}
		go MultiClassCrossValidation(D, nfold, conc)
	}

	f := make([]func(*Options) *Options, nt-1)
	f[0] = func(o *Options) *Options { o.Gamma -= (o.Gamma * fd); return o }
	f[1] = func(o *Options) *Options { o.Gamma += (o.Gamma * fd); return o }

	f[2] = func(o *Options) *Options { o.Lambda -= (o.Lambda * fd); return o }
	f[3] = func(o *Options) *Options { o.Lambda += (o.Lambda * fd); return o }

	f[4] = func(o *Options) *Options { o.SubSample -= (o.SubSample * fd); return o }
	f[5] = func(o *Options) *Options { o.SubSample += (o.SubSample * fd); return o }

	f[6] = func(o *Options) *Options { o.ColSubSample -= (o.ColSubSample * fd); return o }
	f[7] = func(o *Options) *Options { o.ColSubSample += (o.ColSubSample * fd); return o }
	f[8] = func(o *Options) *Options { o.LearningRate -= (o.LearningRate * fd); return o }
	f[9] = func(o *Options) *Options { o.LearningRate += (o.LearningRate * fd); return o }

	f[10] = func(o *Options) *Options { o.Rounds -= int(float64(o.Rounds) * fd); return o }
	f[11] = func(o *Options) *Options { o.Rounds += int(float64(o.Rounds) * fd); return o }

	var i = 0
	var excluded []int
	for _, v := range f {
		if !central && i%2 == 0 {
			i++
			continue
		}
		o := O.Clone()
		o = v(o)
		if err := checkAgainstOptions(o, CVO); err != nil {
			excluded = append(excluded, i)
			continue
		}
		conc := &CVOptions{O: o, Acc: accs[i], Err: errs[i], Ochan: os[i], Conc: true}
		go MultiClassCrossValidation(D, nfold, conc)
		i++
	}

	rec := func(i int) float64 {
		if slices.Contains(excluded, i) {
			return -1000
		}
		err := <-errs[i]
		if err != nil {
			panic(err)
		}
		a := <-accs[i]
		return a
	}
	gm := map[string]float64{"gam": 0, "lam": 0, "ss": 0, "css": 0, "lr": 0, "roun": 0}
	if central {

		pr := func(i, j int, ori float64) float64 {
			r1 := rec(i)
			r2 := rec(j)
			if r1 < -900 || r2 < -900 {
				return 0
			}
			return (r1 - r2) / (2 * ori * fd)
		}

		gm["gam"] = pr(1, 0, O.Gamma)
		gm["lam"] = pr(3, 2, O.Lambda)
		gm["ss"] = pr(5, 4, O.SubSample)
		gm["css"] = pr(7, 6, O.ColSubSample)
		gm["lr"] = pr(9, 8, O.LearningRate)
		gm["rou"] = pr(11, 10, float64(O.Rounds))
	} else {
		curr := rec(last)
		proce := func(i int, ori float64) float64 {
			r := rec(i)
			if r < -900 {
				return 0
			}
			return (r - curr) / (ori * fd)
		}
		gm["gam"] = proce(1, O.Gamma)
		gm["lam"] = proce(3, O.Lambda)
		gm["ss"] = proce(5, O.SubSample)
		gm["css"] = proce(7, O.ColSubSample)
		gm["lr"] = proce(9, O.LearningRate)
		gm["rou"] = proce(11, float64(O.Rounds))
	}
	allzeros := true
	closenough := 0.001
	for k, v := range gm {
		if v < closenough {
			gm[k] = 0.0
		} else {
			allzeros = false
		}
	}
	if allzeros {
		return nil
	}
	reg := func(t float64, allzeros bool) (float64, bool) {
		_ = allzeros //freaking linter.
		return t, false
		/*
			if t <= 0 {
				if !allzeros {
					return 0, false
				}
				return 0, true
			}
			return t, false
		*/
	}
	allzeros = true
	O.Gamma, allzeros = reg(O.Gamma+CVO.Gamma[2]*gm["gam"], allzeros)
	O.Lambda, allzeros = reg(O.Lambda+CVO.Lambda[2]*step*gm["lam"], allzeros)
	O.SubSample, allzeros = reg(O.SubSample+CVO.SubSample[2]*step*gm["ss"], allzeros)
	O.ColSubSample, allzeros = reg(O.ColSubSample+CVO.ColSubSample[2]*step*gm["css"], allzeros)
	O.LearningRate, allzeros = reg(O.LearningRate+CVO.LearningRate[2]*step*gm["lr"], allzeros)
	O.Rounds = O.Rounds + int(float64(CVO.Rounds[2])*step*gm["roun"])
	if O.Rounds <= 0 {
		O.Rounds = 0
	} else {
		allzeros = false
	}
	if checkAgainstOptions(O, CVO) != nil {
		return nil
	}
	return O
}

func setSomeOptionsToMid(o *Options, co *CVGridOptions) *Options {
	av := func(i1, i2 float64) float64 { return (i1 + i2) / 2 }
	avint := func(i1, i2 int) int { return (i1 + i2) / 2 }
	o.Rounds = avint(co.Rounds[0], co.Rounds[1])
	o.LearningRate = av(co.LearningRate[0], co.LearningRate[1])
	o.SubSample = av(co.SubSample[0], co.SubSample[1])
	o.ColSubSample = av(co.ColSubSample[0], co.ColSubSample[1])
	o.Lambda = av(co.Lambda[0], co.Lambda[1])
	//	o.MinChildWeight = av(co.MinChildWeight[0], co.MinChildWeight[1])
	o.Gamma = av(co.Gamma[0], co.Gamma[1])
	// o.MaxDepth = avint(co.MaxDepth[0], co.MaxDepth[1])
	return o
}

// uses 5 gorutines.
func GradientConcCVGrid(data *utils.DataBunch, nfold int, options ...*CVGridOptions) (float64, []float64, *Options, error) {
	var o *CVGridOptions
	if len(options) > 0 && options[0] != nil {
		o = options[0]
	} else {
		o = DefaultXCVGridOptions()
	}
	defaultoptions := DefaultGOptions
	if o.XGB {
		defaultoptions = DefaultXOptions
	}
	var finaloptions *Options
	var accuracies []float64
	/*
		accuracies := make([]float64, 0, 100)
		accs := make([]chan float64, o.NCPUs)
		errs := make([]chan error, o.NCPUs)

		os := make([]chan *Options, o.NCPUs)
		//consider the goroutines used by the GradStep function: 6 if you don't use central differences,
		//10 if you do.

		if o.Central {
			o.NCPUs /= 12
		} else {
			o.NCPUs /= 7
		}
		if o.NCPUs == 0 {
			o.NCPUs = 1
		}
		println("cpus", o.NCPUs) //////////////////
		for i := range o.NCPUs {
			accs[i] = make(chan float64)
			errs[i] = make(chan error)
			os[i] = make(chan *Options)
		}
	*/
	//A bit less hellish than the other function
	bestacc := 0.0
	for cw := o.MinChildWeight[0]; cw <= o.MinChildWeight[1]; cw += o.MinChildWeight[2] {
		for md := o.MaxDepth[0]; md <= o.MaxDepth[1]; md += o.MaxDepth[2] {
			t := defaultoptions()
			t = setSomeOptionsToMid(t, o)
			t.MaxDepth = md
			t.MinChildWeight = cw
			t.XGB = o.XGB
			maxstrikes := 50
			strikes := maxstrikes
			for s := 0; s < o.NSteps; s++ {
				t1 := t
				t = GradStep(t, o, data, o.Step, o.DeltaFraction, nfold, o.Central)
				if t == nil {
					t = t1
					println("nilrecu")
					strikes--
					if strikes == 0 {
						break
					}
					continue
				}
				acc, err := MultiClassCrossValidation(data, 5, &CVOptions{O: t, Conc: false})
				if err != nil {
					return -1, nil, nil, err
				}
				if acc > bestacc {
					accuracies = append(accuracies, acc)
					finaloptions = t
					bestacc = acc
				}
				strikes = maxstrikes
			}
		}

	}
	return bestacc, accuracies, finaloptions, nil
}
