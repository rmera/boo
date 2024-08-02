package boo

import (
	"bufio"
	"fmt"
	"os"
	"testing"

	"github.com/rmera/boo/utils"
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
	O.EarlyStop = 2
	O.Verbose = true
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

func TestFeatures(Te *testing.T) {
	O := DefaultXOptions()
	data, err := utils.DataBunchFromLibSVMFile("tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	b := NewMultiClass(data, O)
	acc := b.Accuracy(data)
	fmt.Printf("Train accuracy: %.3f\n", acc)
	feat, err := b.FeatureImportance()
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("XGBoost:\n", feat.String())

	O = DefaultGOptions()
	b = NewMultiClass(data, O)
	acc = b.Accuracy(data)
	fmt.Printf("Train accuracy: %.3f\n", acc)
	feat, err = b.FeatureImportance()
	if err != nil {
		Te.Error(err)
	}
	fmt.Println("GBoost:\n", feat.String())
}
