package cv

import (
	"fmt"
	"math/rand/v2"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

func setOptionsToMid(o *boo.Options, co *GridOptions) {
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

func checkAgainstOptions(o *boo.Options, co *GridOptions) error {
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

func CurrentValue(O *boo.Options, param string) float64 {
	switch param {
	case "Gamma":
		return O.Gamma
	case "Lambda":
		return O.Lambda
	case "SubSample":
		return O.SubSample
	case "ColSubSample":
		return O.ColSubSample
	case "LearningRate":
		return O.LearningRate
	case "Rounds":
		return float64(O.Rounds)
	default:
		return -1
	}
}

func ParamaterGradStep(D *utils.DataBunch, Op *boo.Options, CVO *GridOptions, param string, step, fractiondelta, currentAccuracy float64, nfold int, central bool, out chan *boo.Options) {
	fd := fractiondelta
	O := Op.Clone()
	type myf func(*boo.Options) *boo.Options

	f := make(map[string][2]myf, 11)
	f["Curr"] = [2]myf{func(o *boo.Options) *boo.Options { return o }, func(o *boo.Options) *boo.Options { return o }}
	f["Gamma"] = [2]myf{func(o *boo.Options) *boo.Options { o.Gamma -= (o.Gamma * fd); return o }, func(o *boo.Options) *boo.Options { o.Gamma += (o.Gamma * fd); return o }}

	f["Lambda"] = [2]myf{func(o *boo.Options) *boo.Options { o.Lambda -= (o.Lambda * fd); return o }, func(o *boo.Options) *boo.Options { o.Lambda += (o.Lambda * fd); return o }}

	f["SubSample"] = [2]myf{func(o *boo.Options) *boo.Options { o.SubSample -= (o.SubSample * fd); return o }, func(o *boo.Options) *boo.Options { o.SubSample += (o.SubSample * fd); return o }}

	f["ColSubSample"] = [2]myf{func(o *boo.Options) *boo.Options { o.ColSubSample -= (o.ColSubSample * fd); return o }, func(o *boo.Options) *boo.Options { o.ColSubSample += (o.ColSubSample * fd); return o }}

	f["LearningRate"] = [2]myf{func(o *boo.Options) *boo.Options { o.LearningRate -= (o.LearningRate * fd); return o }, func(o *boo.Options) *boo.Options { o.LearningRate += (o.LearningRate * fd); return o }}

	f["Rounds"] = [2]myf{func(o *boo.Options) *boo.Options { o.Rounds -= int(float64(o.Rounds) * fd); return o }, func(o *boo.Options) *boo.Options { o.Rounds += int(float64(o.Rounds) * fd); return o }}

	pO := f[param][1](O.Clone())
	pluserr := make(chan error)
	pluso := make(chan *boo.Options)
	plusacc := make(chan float64)
	conc := &Options{O: pO, Acc: plusacc, Err: pluserr, Ochan: pluso, Conc: true}
	go MultiClassCrossValidation(D, nfold, conc)

	var minerr chan error
	var mino chan *boo.Options
	var minacc chan float64

	if central {
		mO := f[param][0](O.Clone())
		if mO.Check() != nil {
			out <- O

		}
		minerr = make(chan error)
		mino = make(chan *boo.Options)
		minacc = make(chan float64)
		conc := &Options{O: mO, Acc: minacc, Err: minerr, Ochan: mino, Conc: true}
		go MultiClassCrossValidation(D, nfold, conc)
	}

	err := <-pluserr
	pacc := <-plusacc
	var der float64
	_ = <-pluso
	if err != nil {
		out <- O
	}
	h := fd * CurrentValue(O, param)
	if !central {
		if h < 0 {
			out <- O
			return
		}
		der = (pacc - currentAccuracy) / h
	} else {
		err = <-minerr
		macc := <-minacc
		_ = <-mino
		if err != nil {
			out <- O
		}
		der = (pacc - macc) / 2 * h
	}
	prev := O.Clone()

	for st := step; st > 0.1*step; st -= st * 0.2 {
		switch param {
		case "Gamma":
			O.Gamma += der * CVO.Gamma[2] * st
		case "Lambda":
			O.Lambda += der * CVO.Lambda[2] * st
		case "SubSample":
			O.SubSample += der * CVO.SubSample[2] * st
		case "ColSubSample":
			O.ColSubSample += der * CVO.ColSubSample[2] * st
		case "LearningRate":
			O.LearningRate += der * CVO.LearningRate[2] * st
		case "Rounds":
			O.Rounds += int(der * float64(CVO.Rounds[2]) * st)
		default:
			out <- O
			return
		}
		if checkAgainstOptions(O, CVO) == nil {
			break
		}
		O = prev.Clone() //the previous check failed
	}
	if checkAgainstOptions(O, CVO) != nil {
		O = prev
	}
	out <- O
	return
}

func GradStep(Ori *boo.Options, CVO *GridOptions, D *utils.DataBunch, step, fractiondelta float64, nfold int, central bool, out chan *boo.Options) *boo.Options {
	O := Ori
	if out != nil {
		O = Ori.Clone()
	}
	var curracc float64
	if !central {
		var err error
		conc := &Options{O: O, Acc: nil, Err: nil, Ochan: nil, Conc: false}
		curracc, err = MultiClassCrossValidation(D, nfold, conc)
		if err != nil {
			return Ori
		}
	}
	gm := map[string]chan *boo.Options{"Gamma": make(chan *boo.Options), "Lambda": make(chan *boo.Options), "SubSample": make(chan *boo.Options), "ColSubSample": make(chan *boo.Options), "LearningRate": make(chan *boo.Options), "Rounds": make(chan *boo.Options)}

	for k, v := range gm {
		go ParamaterGradStep(D, O, CVO, k, step, fractiondelta, curracc, nfold, central, v)
	}
	O.Gamma = (<-gm["Gamma"]).Gamma
	O.Lambda = (<-gm["Lambda"]).Lambda
	O.SubSample = (<-gm["SubSample"]).SubSample
	O.LearningRate = (<-gm["LearningRate"]).LearningRate
	O.Rounds = (<-gm["Rounds"]).Rounds
	O.ColSubSample = (<-gm["ColSubSample"]).ColSubSample
	if out != nil {
		out <- O
	}
	return O

}

func setSomeOptionsToMid(o *boo.Options, co *GridOptions) *boo.Options {
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
func GradientConcCVGrid(data *utils.DataBunch, nfold int, options ...*GridOptions) (float64, []float64, *boo.Options, error) {
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
	var accuracies []float64
	//A bit less hellish than the other function
	bestacc := 0.0
	for cw := o.MinChildWeight[0]; cw <= o.MinChildWeight[1]; cw += o.MinChildWeight[2] {
		if o.Verbose {
			fmt.Println("ChildrenWeight:", cw)
		}
		for md := o.MaxDepth[0]; md <= o.MaxDepth[1]; md += o.MaxDepth[2] {
			t := defaultoptions()
			t = setSomeOptionsToMid(t, o)
			t.MaxDepth = md
			t.MinChildWeight = cw
			t.XGB = o.XGB
			tprev := t.Clone()
			CompareAccs := func(t, tprev *boo.Options) (*boo.Options, error) {
				acc, err := MultiClassCrossValidation(data, 5, &Options{O: t, Conc: false})
				if err != nil {
					return nil, err
				}
				if len(accuracies) > 0 && acc < accuracies[len(accuracies)-1] {
					//			println("rejected step")
					return fuzzOptions(tprev), nil

				}
				//		println("new (might not be best) accuracy!", acc)
				if acc > bestacc {
					if o.Verbose {
						fmt.Println("New best accuracy: ", acc, t)
					}
					accuracies = append(accuracies, acc)
					finaloptions = t
					bestacc = acc
				}
				return t, nil
			}
			t, err := CompareAccs(t, tprev)
			if err != nil {
				return -1, nil, nil, err
			}

			maxstrikes := 3
			strikes := 0
			for s := 0; s < o.NSteps; s++ {
				t1 := t.Clone()
				t = GradStep(t, o, data, o.Step, o.DeltaFraction, nfold, o.Central, nil)
				if t.Equal(t1) {
					strikes++
					t = fuzzOptions(t, 0.2) //switch to 0.1
					if strikes == maxstrikes {
						strikes = 0
						break
					}
					continue
				}
				t, err = CompareAccs(t, t1)
				if err != nil {
					return -1, nil, nil, err
				}
				strikes = 0
			}
		}
	}
	return bestacc, accuracies, finaloptions, nil
}

// returns f plus or minus up to fuzzperc*f
func fuzz(f, fuzzperc float64) float64 {
	fu := rand.Float64() * fuzzperc
	sign := 1.0
	if rand.IntN(2) < 1 {
		sign = -1
	}
	return f + sign*fu*f
}

func fuzzOptions(O *boo.Options, fuzzperc ...float64) *boo.Options {
	f := 0.1
	if len(fuzzperc) > 0 && fuzzperc[0] > 0 {
		f = fuzzperc[0]
	}
	O.Rounds = int(fuzz(float64(O.Rounds), f))
	O.SubSample = fuzz(O.SubSample, f)
	O.ColSubSample = fuzz(O.ColSubSample, f)
	O.Lambda = fuzz(O.Lambda, f)
	O.Gamma = fuzz(O.Gamma, f)
	O.LearningRate = fuzz(O.LearningRate, f)
	O.BaseScore = fuzz(O.BaseScore, f)
	return O
}
