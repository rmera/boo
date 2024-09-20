package cv

import (
	"fmt"
	"testing"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

func TestCrossValXBoostEarlyStop(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}

	testdata, err := utils.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}

	O := new(boo.Options)
	O.XGB = true
	O.Rounds = 500
	O.SubSample = 0.9
	O.Lambda = 1.1
	O.Gamma = 0.1
	O.MinChildWeight = 2
	O.MaxDepth = 3
	O.LearningRate = 0.2
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.EarlyStop = 5
	O.Verbose = true
	O.Loss = &utils.SQErrLoss{}
	//87%, 50 r/3 md/0.200 lr/0.900 ss/0.500 bs/0.100 gam/1.100 lam/2.000 mcw/
	acc, err := MultiClassCrossValidation(data, 8, &Options{O: O, Conc: false})
	fmt.Println("Crossvalidation best accuracy:", acc)
	b := boo.NewMultiClass(data, O)
	acc = b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)
}

func TestCrossValXBoost(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}

	testdata, err := utils.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}

	O := new(boo.Options)
	O.XGB = true
	O.Rounds = 50
	O.SubSample = 0.9
	O.Lambda = 1.1
	O.Gamma = 0.1
	O.MinChildWeight = 2
	O.MaxDepth = 3
	O.LearningRate = 0.2
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.Loss = &utils.SQErrLoss{}
	//87%, 50 r/3 md/0.200 lr/0.900 ss/0.500 bs/0.100 gam/1.100 lam/2.000 mcw/
	acc, err := MultiClassCrossValidation(data, 8, &Options{O: O, Conc: false})
	fmt.Println("Crossvalidation best accuracy:", acc)
	b := boo.NewMultiClass(data, O)
	acc = b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)
}

func TestCrossValGBoost(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	testdata, err := utils.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}
	O := new(boo.Options)
	O.XGB = false
	O.Rounds = 50
	O.MinChildWeight = 2
	O.MaxDepth = 3
	O.LearningRate = 0.2
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.Loss = &utils.MSELoss{}
	cv := &Options{O: O, Conc: false}
	//87%, 50 r/3 md/0.200 lr/0.900 ss/0.500 bs/0.100 gam/1.100 lam/2.000 mcw/
	acc, err := MultiClassCrossValidation(data, 8, cv)
	fmt.Println("Crossvalidation best accuracy:", acc)
	b := boo.NewMultiClass(data, O)
	acc = b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)
}

func TestCrossValGradGrid(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	testdata, err := utils.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}
	o := DefaultXGridOptions()
	o.Rounds = [3]int{50, 200, 50}
	o.MaxDepth = [3]int{3, 5, 1}
	o.Lambda = [3]float64{0, 20, 1}
	o.LearningRate = [3]float64{0.1, 0.3, 0.1}
	o.Gamma = [3]float64{0.0, 0.9, 0.3}
	o.SubSample = [3]float64{0.1, 0.9, 0.3}
	o.ColSubSample = [3]float64{0.1, 0.9, 0.3}
	o.MinChildWeight = [3]float64{2, 6, 2}
	o.DeltaFraction = 0.01
	o.Central = false //////////////////////////
	o.Verbose = true
	o.NSteps = 100
	o.NCPUs = 4
	o.Step = 0.05

	bestacc, accuracies, best, err := GradientGrid(data, 5, o)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Crossvalidation best accuracy:", bestacc)
	fmt.Printf("With %d rounds, %d maxdepth and %.3f learning rate\n", best.Rounds, best.MaxDepth, best.LearningRate)
	fmt.Println(best)
	fmt.Println("All accuracies:", accuracies)

	b := boo.NewMultiClass(data, best)
	acc := b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)
}

func TestGradStep(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	o := DefaultXGridOptions()
	o.Rounds = [3]int{10, 110, 50}
	o.MaxDepth = [3]int{3, 5, 1}
	o.Lambda = [3]float64{0, 10, 1}
	o.LearningRate = [3]float64{0.1, 0.6, 0.1}
	o.Gamma = [3]float64{0.0, 0.9, 0.1}
	o.SubSample = [3]float64{0.1, 1, 0.1}
	o.ColSubSample = [3]float64{0.1, 1, 0.1}
	o.MinChildWeight = [3]float64{3, 5, 2}
	o.Central = false //////////////////////////
	o.Verbose = true
	o.NCPUs = 4
	o.Step = 0.05
	op := boo.DefaultOptions()
	op = setSomeOptionsToMid(op, o)
	oprev := op.Clone()
	fmt.Println("Will start with the grad steps")
	o.DeltaFraction = 0.01
	var acc = 0.0
	for i := 0; i < 50; i++ {
		fmt.Println("A step will run", op)
		op = GradStep(op, o, data, o.Step, o.DeltaFraction, 5, o.Central, nil)
		fmt.Println("A grad step ran")
		if op == nil {
			fmt.Println("got nil  option from grad step")
			op = oprev.Clone()
			op = oprev.Clone()
		} else {
			acc, err = MultiClassCrossValidation(data, 5, &Options{O: op, Conc: false})
			if err != nil {
				Te.Error(err)
			}
			oprev = op.Clone()
		}
		fmt.Println("accuracy", acc)
	}

	acc, err = MultiClassCrossValidation(data, 5, &Options{O: op, Conc: false})
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Accuracy", acc)
	fmt.Println("Options after the step", op)

}

func TestCrossValXGBoostGrid(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	testdata, err := utils.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}
	o := DefaultXGridOptions()
	o.Rounds = [3]int{5, 30, 5}
	o.MaxDepth = [3]int{3, 5, 1}
	o.LearningRate = [3]float64{0.05, 0.3, 0.1}
	o.SubSample = [3]float64{0.8, 0.9, 0.1}
	o.MinChildWeight = [3]float64{2, 6, 2}
	o.Verbose = true
	o.NCPUs = 2
	o.WriteBest = true
	bestacc, accuracies, best, err := Grid(data, 8, o)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Crossvalidation best accuracy:", bestacc)
	fmt.Printf("With %d rounds, %d maxdepth and %.3f learning rate\n", best.Rounds, best.MaxDepth, best.LearningRate)
	fmt.Println("All accuracies:", accuracies)

	b := boo.NewMultiClass(data, best)
	acc := b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)
}

func BenchmarkCrossValXGBoostGrid(b *testing.B) {
	data, err := utils.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		b.Error(err)
	}
	o := DefaultXGridOptions()
	o.Rounds = [3]int{20, 30, 10}
	o.MaxDepth = [3]int{7, 9, 1}
	//	o.EarlyStop = 300
	o.LearningRate = [3]float64{0.1, 0.2, 0.1}
	o.SubSample = [3]float64{0.8, 0.9, 0.1}
	o.MinChildWeight = [3]float64{1, 1, 1}
	o.Verbose = true
	o.NCPUs = 2
	for i := 0; i < b.N; i++ {
		_, _, _, err := Grid(data, 5, o)
		if err != nil {
			b.Error(err)
		}
	}
}

func TestConcCrossValGGBoostGrid(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	testdata, err := utils.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}
	o := DefaultGGridOptions()
	o.Rounds = [3]int{5, 10, 5}
	o.MaxDepth = [3]int{3, 5, 1}
	o.LearningRate = [3]float64{0.1, 0.3, 0.1}
	o.MinChildWeight = [3]float64{2, 6, 2}
	o.Verbose = true
	o.NCPUs = 2
	bestacc, accuracies, best, err := Grid(data, 8, o)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Crossvalidation best accuracy:", bestacc)
	fmt.Printf("With %d rounds, %d maxdepth and %.3f learning rate\n", best.Rounds, best.MaxDepth, best.LearningRate)
	fmt.Println("All accuracies:", accuracies)

	b := boo.NewMultiClass(data, best)
	acc := b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)
}

func TestHybridGradGrid(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	testdata, err := utils.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}
	o := DefaultXGridOptions()
	o.Rounds = [3]int{50, 550, 100}
	o.MaxDepth = [3]int{3, 5, 1}
	o.Lambda = [3]float64{0, 20, 5}
	o.LearningRate = [3]float64{0.1, 0.6, 0.1}
	o.Gamma = [3]float64{0.0, 0.9, 0.1}
	o.SubSample = [3]float64{0.1, 0.9, 0.1}
	o.ColSubSample = [3]float64{0.1, 0.9, 0.1}
	o.MinChildWeight = [3]float64{2, 6, 2}
	o.DeltaFraction = 0.01
	o.Central = false //////////////////////////
	o.Verbose = true
	o.NSteps = 100
	o.NCPUs = 4
	o.Step = 0.05
	o.WriteBest = true

	bestacc, accuracies, best, err := HybridGradientGrid(data, 5, o)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Crossvalidation best accuracy:", bestacc)
	fmt.Printf("With %d rounds, %d maxdepth and %.3f learning rate\n", best.Rounds, best.MaxDepth, best.LearningRate)
	fmt.Println(best)
	fmt.Println("All accuracies:", accuracies)

	b := boo.NewMultiClass(data, best)
	acc := b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)
}
