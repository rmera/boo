package learn

import (
	"bufio"
	"fmt"
	"os"
	"testing"

	"github.com/rmera/chemlearn/utils"
	"gonum.org/v1/gonum/mat"
)

func TestGTree(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("tests/traineasy.svm", true)
	if err != nil {
		Te.Error(err)
	}
	opts := DefaultGTreeOptions()
	opts.MaxDepth = 10
	opts.MinChildWeight = 1
	targets, names := data.OHELabels()
	target := utils.DenseCol(targets, 0)
	fmt.Println("Prob for label", names[0], utils.PrintDenseMatrix(targets), target.RawRowView(0))
	opts.Y = target.RawRowView(0)
	tree := NewTree(data.Data, opts)
	fmt.Println("predictions", tree.Predict(data.Data, nil))
	fmt.Println(tree.Print(""))
	jtree, _, err := utils.JSONTree(tree)
	if err != nil {
		Te.Error(err)
	}
	for _, v := range jtree {
		fmt.Println(string(v))
	}
}

func TestSubSample(Te *testing.T) {
	ntests := 100
	tot := 10
	prob := 0.5
	var i int = 0
	nsamples := 0
	for i = 0; i < ntests; i++ {
		ss := SubSample(tot, prob)
		fmt.Println(i+1, ": samples out of total", 30, "elements", ss, len(ss))
		nsamples += len(ss)
	}
	fmt.Println("sampled an average of", float64(nsamples)/float64(ntests), "From the total", tot, "samples in", ntests, "tests")

}

func TestColSubSampling(Te *testing.T) {
	o := new(TreeOptions)
	o.MinChildWeight = 1
	o.Lambda = 1
	o.Gamma = 0
	o.ColSampleByNode = 1
	o.AllowedColumns = []int{0, 5, 6, 7}
	o.MaxDepth = 4
	o.Debug = true
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	ohelabels, _ := data.OHELabels()
	r, c := ohelabels.Dims()
	rawPred := mat.NewDense(r, c, nil)
	for k := 0; k < 3; k++ {
		fmt.Println("New Tree!")
		kthlabelvector := utils.DenseCol(ohelabels, k)
		kthpreds := utils.DenseCol(rawPred, k)
		mse := &utils.SQErrLoss{}
		grads := mse.Gradients(kthlabelvector, kthpreds, nil)
		hess := mse.Hessian(kthpreds, nil)
		o.Gradients = grads.RawRowView(0)
		o.Hessian = hess.RawRowView(0)
		o.Y = kthlabelvector.RawRowView(0)
		tree := NewTree(data.Data, o)
		fmt.Println(tree.Print("", data.Keys))
	}
}

func TestRowSubSampling(Te *testing.T) {
	o := new(TreeOptions)
	o.MinChildWeight = 1
	o.Lambda = 1
	o.Gamma = 0
	o.ColSampleByNode = 1
	o.Indexes = []int{0, 5, 6, 7}
	o.MaxDepth = 4
	o.Debug = true
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	ohelabels, _ := data.OHELabels()
	r, c := ohelabels.Dims()
	rawPred := mat.NewDense(r, c, nil)
	for k := 0; k < 3; k++ {
		fmt.Println("New Tree!")
		kthlabelvector := utils.DenseCol(ohelabels, k)
		kthpreds := utils.DenseCol(rawPred, k)
		mse := &utils.SQErrLoss{}
		grads := mse.Gradients(kthlabelvector, kthpreds, nil)
		hess := mse.Hessian(kthpreds, nil)
		o.Gradients = grads.RawRowView(0)
		o.Hessian = hess.RawRowView(0)
		o.Y = kthlabelvector.RawRowView(0)
		tree := NewTree(data.Data, o)
		fmt.Println(tree.Print("", data.Keys))
	}
}

func TestXTree(Te *testing.T) {
	o := new(TreeOptions)
	o.MinChildWeight = 1
	o.Lambda = 1
	o.Gamma = 0
	o.ColSampleByNode = 1
	o.MaxDepth = 3
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	ohelabels, _ := data.OHELabels()
	r, c := ohelabels.Dims()
	rawPred := mat.NewDense(r, c, nil)
	for k := 0; k < 3; k++ {
		fmt.Println("New Tree!")
		kthlabelvector := utils.DenseCol(ohelabels, k)
		kthpreds := utils.DenseCol(rawPred, k)
		mse := &utils.SQErrLoss{}
		grads := mse.Gradients(kthlabelvector, kthpreds, nil)
		hess := mse.Hessian(kthpreds, nil)
		o.Gradients = grads.RawRowView(0)
		o.Hessian = hess.RawRowView(0)
		o.Y = kthlabelvector.RawRowView(0)
		tree := NewTree(data.Data, o)
		fmt.Println(tree.Print("", data.Keys))
		jtree, _, err := utils.JSONTree(tree)
		if err != nil {
			Te.Error(err)
		}
		for _, v := range jtree {
			fmt.Println(string(v))
		}
	}
}

func TestXGBoost(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	O := new(Options)
	O.Rounds = 500
	O.SubSample = 0.8
	O.Lambda = 1.5
	O.MinChildWeight = 3
	O.MaxDepth = 3
	O.LearningRate = 0.3
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.Loss = &utils.SQErrLoss{}

	boosted := NewMultiClass(data, O)
	fmt.Println("train set accuracy", boosted.Accuracy(data))
}

func TestGBoost(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	o := DefaultGOptions()
	boosted := NewMultiClass(data, o)
	fmt.Println("test set accuracy", boosted.Accuracy(data))
	jtest := newjsonTester()
	JSONMultiClass(boosted, "softmaxDense", jtest)
	strs := jtest.Str
	for _, v := range strs {
		fmt.Println(v)
	}
}

func TestXJSON(Te *testing.T) {
	O := new(Options)
	O.Rounds = 20
	O.SubSample = 0.9
	O.Lambda = 0.7
	O.Gamma = 0.5
	O.MinChildWeight = 3
	O.MaxDepth = 6
	O.LearningRate = 0.46
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.Loss = &utils.SQErrLoss{}
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	acc, err := MultiClassCrossValidation(data, 5, &CVOptions{O: O, Conc: false})
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Crossvalidation accuracy:", acc)
	boosted := NewMultiClass(data, O)
	fmt.Println("train set accuracy", boosted.Accuracy(data))
	os.Remove("tests/xgbmodel.json")
	f, err := os.Create("tests/xgbmodel.json")
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
	f, err = os.Open("tests/xgbmodel.json")
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

func TestGJSON(Te *testing.T) {
	O := new(Options)
	O.Rounds = 20
	O.MinChildWeight = 3
	O.MaxDepth = 6
	O.LearningRate = 0.46
	O.TreeMethod = "exact"
	O.Loss = &utils.SQErrLoss{}

	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	acc, err := MultiClassCrossValidation(data, 5, &CVOptions{O: O, Conc: false})
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Crossvalidation accuracy:", acc)
	boosted := NewMultiClass(data, O)
	fmt.Println("train set accuracy", boosted.Accuracy(data))
	os.Remove("tests/gbmodel.json")
	f, err := os.Create("tests/gbmodel.json")
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
	f, err = os.Open("tests/gbmodel.json")
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

func TestCrossValXBoost(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}

	testdata, err := utils.DataBunchFromLibSVMFile("tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}

	O := new(Options)
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
	acc, err := MultiClassCrossValidation(data, 8, &CVOptions{O: O, Conc: false})
	fmt.Println("Crossvalidation best accuracy:", acc)
	b := NewMultiClass(data, O)
	acc = b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)
}

func TestCrossValGBoost(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	testdata, err := utils.DataBunchFromLibSVMFile("tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}
	O := new(Options)
	O.XGB = false
	O.Rounds = 50
	O.MinChildWeight = 2
	O.MaxDepth = 3
	O.LearningRate = 0.2
	O.BaseScore = 0.5
	O.TreeMethod = "exact"
	O.Loss = &utils.MSELoss{}
	cv := &CVOptions{O: O, Conc: false}
	//87%, 50 r/3 md/0.200 lr/0.900 ss/0.500 bs/0.100 gam/1.100 lam/2.000 mcw/
	acc, err := MultiClassCrossValidation(data, 8, cv)
	fmt.Println("Crossvalidation best accuracy:", acc)
	b := NewMultiClass(data, O)
	acc = b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)
}

func TestCrossValGradGrid(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	testdata, err := utils.DataBunchFromLibSVMFile("tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}
	o := DefaultXCVGridOptions()
	o.Rounds = [3]int{50, 500, 10}
	o.MaxDepth = [3]int{3, 5, 1}
	o.Lambda = [3]float64{0, 20, 1}
	o.LearningRate = [3]float64{0.1, 0.6, 0.1}
	o.Gamma = [3]float64{0.0, 0.9, 0.1}
	o.SubSample = [3]float64{0.1, 0.9, 0.1}
	o.ColSubSample = [3]float64{0.1, 0.9, 0.1}
	o.MinChildWeight = [3]float64{2, 6, 2}
	o.DeltaFraction = 0.01
	o.Central = true //////////////////////////
	o.Verbose = true
	o.NSteps = 100
	o.NCPUs = 4
	o.Step = 0.05 / 2

	bestacc, accuracies, best, err := GradientConcCVGrid(data, 5, o)
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Crossvalidation best accuracy:", bestacc)
	fmt.Printf("With %d rounds, %d maxdepth and %.3f learning rate\n", best.Rounds, best.MaxDepth, best.LearningRate)
	fmt.Println(best)
	fmt.Println("All accuracies:", accuracies)

	b := NewMultiClass(data, best)
	acc := b.Accuracy(testdata)
	fmt.Printf("Test accuracy: %.3f\n", acc)
}

func TestGradStep(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	o := DefaultXCVGridOptions()
	o.Rounds = [3]int{150, 400, 10}
	o.MaxDepth = [3]int{3, 5, 1}
	o.Lambda = [3]float64{0, 20, 1}
	o.LearningRate = [3]float64{0.1, 0.6, 0.1}
	o.Gamma = [3]float64{0.0, 0.9, 0.1}
	o.SubSample = [3]float64{0.1, 1, 0.1}
	o.ColSubSample = [3]float64{0.1, 1, 0.1}
	o.MinChildWeight = [3]float64{2, 6, 2}
	o.Central = false //////////////////////////
	o.Verbose = true
	o.NCPUs = 4
	o.Step = 0.05
	op := DefaultOptions()
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
			acc, err = MultiClassCrossValidation(data, 5, &CVOptions{O: op, Conc: false})
			if err != nil {
				Te.Error(err)
			}
			oprev = op.Clone()
		}
		fmt.Println("accuracy", acc)
	}

	acc, err = MultiClassCrossValidation(data, 5, &CVOptions{O: op, Conc: false})
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("Accuracy", acc)
	fmt.Println("Options after the step", op)

}

func TestCrossValXGBoostGrid(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	testdata, err := utils.DataBunchFromLibSVMFile("tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}
	o := DefaultXCVGridOptions()
	o.Rounds = [3]int{5, 10, 5}
	o.MaxDepth = [3]int{3, 5, 1}
	o.LearningRate = [3]float64{0.1, 0.3, 0.1}
	o.SubSample = [3]float64{0.8, 0.9, 0.1}
	o.MinChildWeight = [3]float64{2, 6, 2}
	o.Verbose = true
	o.NCPUs = 2
	bestacc, accuracies, best, err := ConcCVGrid(data, 8, o)
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

func TestConcCrossValGGBoostGrid(Te *testing.T) {
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	testdata, err := utils.DataBunchFromLibSVMFile("tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}
	o := DefaultGCVGridOptions()
	o.Rounds = [3]int{5, 10, 5}
	o.MaxDepth = [3]int{3, 5, 1}
	o.LearningRate = [3]float64{0.1, 0.3, 0.1}
	o.MinChildWeight = [3]float64{2, 6, 2}
	o.Verbose = true
	o.NCPUs = 2
	bestacc, accuracies, best, err := ConcCVGrid(data, 8, o)
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

func TestFeatures(Te *testing.T) {
	O := DefaultXOptions()
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	acc, err := MultiClassCrossValidation(data, 5, &CVOptions{O: O, Conc: false})
	if err != nil {
		Te.Error(err)
	}
	b := NewMultiClass(data, O)
	acc = b.Accuracy(data)
	fmt.Printf("Train accuracy: %.3f\n", acc)
	feat, err := b.FeatureImportance()
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("XGBoost:\n", feat.String())

	O = DefaultGOptions()
	acc, err = MultiClassCrossValidation(data, 5, &CVOptions{O: O, Conc: false})
	if err != nil {
		Te.Error(err)
	}
	b = NewMultiClass(data, O)
	acc = b.Accuracy(data)
	fmt.Printf("Train accuracy: %.3f\n", acc)
	feat, err = b.FeatureImportance()
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("GBoost:\n", feat.String())
}
