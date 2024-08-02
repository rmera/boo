package main

import (
	"fmt"

	"github.com/rmera/boo"
	"github.com/rmera/boo/cv"
	"github.com/rmera/boo/utils"
)

func main() {
	data, err := utils.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		panic(err)
	}
	O := boo.DefaultOptions()
	O.Rounds = 100
	O.ColSubSample = 0.8
	O.EarlyStop = 10
	O.Verbose = true
	O.Loss = &utils.SQErrLoss{}

	boosted := boo.NewMultiClass(data, O)
	fmt.Println("train set accuracy", boosted.Accuracy(data))

	o := cv.DefaultXGridOptions()
	o.Rounds = [3]int{5, 30, 5}
	o.MaxDepth = [3]int{3, 4, 1}
	o.LearningRate = [3]float64{0.1, 0.3, 0.1}
	o.SubSample = [3]float64{0.8, 0.9, 0.1}
	o.MinChildWeight = [3]float64{2, 6, 2}
	o.Verbose = true
	o.NCPUs = 2
	bestacc, accuracies, best, err := cv.Grid(data, 8, o)
	if err != nil {
		panic(err)
	}
	fmt.Println("Crossvalidation best accuracy:", bestacc)
	fmt.Printf("With %d rounds, %d maxdepth and %.3f learning rate\n", best.Rounds, best.MaxDepth, best.LearningRate)
	fmt.Println("All accuracies:", accuracies)

	//You probably want to expand the search space for this one.
	bestacc, accuracies, best, err = cv.GradientGrid(data, 5, o)
	if err != nil {
		panic(err)
	}
	fmt.Println("Crossvalidation (grad) best accuracy:", bestacc)
	fmt.Printf("With %d rounds, %d maxdepth and %.3f learning rate\n", best.Rounds, best.MaxDepth, best.LearningRate)
	fmt.Println(best)
	fmt.Println("All accuracies:", accuracies)

}
