package xgb

import (
	"bufio"
	"fmt"
	"os"
	"testing"

	"github.com/rmera/learn"
	"gonum.org/v1/gonum/mat"
)

func TestTree(Te *testing.T) {

	o := new(TreeOptions)
	o.MinChildWeight = 1
	o.RegLambda = 1
	o.Gamma = 0
	o.ColSampleByNode = 1
	o.MaxDepth = 3

	data, err := learn.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	ohelabels, _ := data.OHELabels()
	r, c := ohelabels.Dims()
	rawPred := mat.NewDense(r, c, nil)
	for k := 0; k < 3; k++ {
		fmt.Println("New Tree!")
		kthlabelvector := learn.DenseCol(ohelabels, k)
		kthpreds := learn.DenseCol(rawPred, k)
		mse := &learn.SQErrLoss{}
		ngrads := mse.Gradients(kthlabelvector, kthpreds, nil)
		hess := mse.Hessian(kthpreds, nil)
		tree := NewTree(data.Data, kthlabelvector.RawRowView(0), ngrads.RawRowView(0), hess.RawRowView(0), o)
		fmt.Println(tree.Print("", data.Keys))
	}

}

func TestXGBoost(Te *testing.T) {
	data, err := learn.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	O := new(Options)
	O.Rounds = 500
	O.SubSample = 0.8
	O.RegLambda = 1.5
	O.MinChildWeight = 3
	O.MaxDepth = 3
	O.LearningRate = 0.3
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.Loss = &learn.SQErrLoss{}

	boosted := NewMultiClass(data, O)
	fmt.Println("train set accuracy", boosted.Accuracy(data))
}

func TestJSON(Te *testing.T) {
	//	ctm := &wql{}
	O := new(Options)
	O.Rounds = 20
	O.SubSample = 0.9
	O.RegLambda = 0.7
	O.Gamma = 0.5
	O.MinChildWeight = 3
	O.MaxDepth = 6
	O.LearningRate = 0.46
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.Loss = &learn.SQErrLoss{}

	data, err := learn.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	acc, err := MultiClassCrossValidation(data, 5, O)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Crossvalidation accuracy:", acc)
	boosted := NewMultiClass(data, O)
	fmt.Println("train set accuracy", boosted.Accuracy(data))
	os.Remove("../tests/xgbmodel.json")
	f, err := os.Create("../tests/xgbmodel.json")
	if err != nil {
		Te.Error(err)
	}
	bf := bufio.NewWriter(f)
	err = JSONMultiClass(boosted, "softmax", bf)
	if err != nil {
		Te.Error(err)
	}
	bf.Flush()
	f.Close()
	f, err = os.Open("../tests/xgbmodel.json")
	defer f.Close()
	if err != nil {
		Te.Error(err)
	}
	b := bufio.NewReader(f)
	m, err := UnJSONMultiClass(b)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("train set accuracy of the recovered object", m.Accuracy(data))
}

func TestCrossValGBoost(Te *testing.T) {
	data, err := learn.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}

	testdata, err := learn.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}

	O := new(Options)
	O.Rounds = 50
	O.SubSample = 0.9
	O.RegLambda = 1.1
	O.Gamma = 0.1
	O.MinChildWeight = 2
	O.MaxDepth = 3
	O.LearningRate = 0.2
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.Loss = &learn.SQErrLoss{}

	//87%, 50 r/3 md/0.200 lr/0.900 ss/0.500 bs/0.100 gam/1.100 lam/2.000 mcw/

	acc, err := MultiClassCrossValidation(data, 8, O)

	fmt.Println("Crossvalidation best accuracy:", acc)

	b := NewMultiClass(data, O)
	acc = b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)

}

func TTestConcCrossValGBoostGrid(Te *testing.T) {
	data, err := learn.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}

	testdata, err := learn.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}

	o := DefaultCVGridOptions()
	o.Rounds = [3]int{50, 1000, 100}
	o.MaxDepth = [3]int{3, 5, 1}
	o.LearningRate = [3]float64{0.05, 0.4, 0.05}
	o.SubSample = [3]float64{0.7, 0.9, 0.1}
	o.MinChildWeight = [3]float64{2, 5, 1}
	o.NCPUs = 2

	bestacc, accuracies, best, err := ConcCVGrid(data, 8, true, o)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Crossvalidation best accuracy:", bestacc)
	fmt.Printf("With %d rounds, %d maxdepth and %.3f learning rate\n", best.Rounds, best.MaxDepth, best.LearningRate)
	fmt.Println("All accuracies:", accuracies)

	b := NewMultiClass(data, best)
	acc := b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)

}
