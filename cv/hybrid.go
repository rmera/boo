package cv

import (
	"fmt"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

// uses 5 gorutines.
func HybridGradientGrid(data *utils.DataBunch, nfold int, options ...*GridOptions) (float64, []float64, *boo.Options, error) {
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
		for rounds := o.Rounds[0]; rounds <= o.Rounds[1]; rounds += o.Rounds[2] {
			if o.Verbose {
				fmt.Println("Rounds: ", rounds, "MinChildWeight:", cw)
			}
			for md := o.MaxDepth[0]; md <= o.MaxDepth[1]; md += o.MaxDepth[2] {
				for lam := o.Lambda[0]; lam <= o.Lambda[1]; lam += o.Lambda[2] {
					o2 := o.Clone()
					or := o.Rounds
					cr := int(float64(or[2]) * 0.1)

					o2.Rounds = [3]int{rounds - or[2], rounds + or[2], cr}
					lram := float64(lam) * 0.1
					oldlam := o.Lambda
					o2.Lambda = [3]float64{lam - oldlam[2], lam + oldlam[2], lram}

					t := defaultoptions()
					t = setSomeOptionsToMid(t, o2)
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
							return fuzzOptions(tprev), nil
						}
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
		}
	}
	return bestacc, accuracies, finaloptions, nil
}
