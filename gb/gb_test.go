package gb

import (
	"bufio"
	"fmt"
	"os"
	"testing"

	"github.com/rmera/learn"
)

func TestLibSVM(Te *testing.T) {
	data, err := learn.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println(data.String())
	fmt.Println(data.LibSVM())

}

func TestTree(Te *testing.T) {
	data, err := learn.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	opts := DefaultTreeOptions()
	opts.MaxDepth = 10
	opts.MinChildWeight = 1
	targets, names := data.OHELabels()
	target := learn.DenseCol(targets, 0)
	fmt.Println("Prob for label", names[0], learn.PrintDenseMatrix(targets), target.RawRowView(0))

	tree := NewTree(data.Data, target.RawRowView(0), opts)

	fmt.Println("predictions", tree.Predict(data.Data, nil))
	fmt.Println(tree.Print(""))

}

func TestGBoost(Te *testing.T) {
	data, err := learn.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	boosted := NewMultiClass(data)
	fmt.Println("test set accuracy", boosted.Accuracy(data))
}

func TestCrossValGBoost(Te *testing.T) {
	data, err := learn.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}

	acc, err := MultiClassCrossValidation(data, 5)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Crossvalidation accuracy:", acc)

}

func TestCrossValGBoostGrid(Te *testing.T) {
	data, err := learn.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}

	testdata, err := learn.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}

	fmt.Println("Both sets read!")
	opts := DefaultCVGridOptions()
	opts.Verbose = true
	bestacc, accuracies, best, err := CVGrid(data, 5, opts)
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

func TTestCrossValGBGridOfGrids(Te *testing.T) {
	data, err := learn.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}

	testdata, err := learn.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}

	fmt.Println("Both sets read!")

	accs, opts, err := CVGridofGrids(data, 5, 8)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Crossvalidation best accuracies:", accs[:9])
	fmt.Printf("Best ones with:")
	for _, v := range opts[:9] {
		fmt.Printf("%s ", v.String())
	}

	b := NewMultiClass(data, opts[0])
	acc := b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)

}

func TestJSON(Te *testing.T) {
	//	ctm := &wql{}
	O := new(Options)
	O.Rounds = 20
	O.MinChildWeight = 3
	O.MaxDepth = 6
	O.LearningRate = 0.46
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
	os.Remove("../tests/gbmodel.json")
	f, err := os.Create("../tests/gbmodel.json")
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
	f, err = os.Open("../tests/gbmodel.json")
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
