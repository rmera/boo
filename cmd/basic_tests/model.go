package main

import (
	"bufio"
	"fmt"
	"os"

	"github.com/rmera/boo"
	"github.com/rmera/boo/confu"
	"github.com/rmera/boo/utils"
)

func main() {
	f, err := os.Open(os.Args[1])
	defer f.Close()
	if err != nil {
		panic(err)
	}
	b := bufio.NewReader(f)
	m, err := boo.UnJSONMultiClass(b)
	if err != nil {
		panic(err)
	}
	data, err := utils.DataBunchFromLibSVMFile(os.Args[2], true)
	if err != nil {
		panic(err)
	}
	fmt.Println("Labels", data.Labels) ///////
	fmt.Println("train set accuracy of the recovered object", m.Accuracy(data))

	feat, err := m.FeatureImportance()
	if err != nil {
		panic(err)
	}
	fmt.Println("XGBoost:\n", feat.String())

	namemap, err := confu.ReadNameMap(os.Args[3])
	if err != nil {
		panic(err)
	}
	C := confu.MCConfusions(m, data)
	topn := C.PrintTopN(5, namemap)
	fmt.Printf(topn + "\n")
	fmt.Println(C.Actual, C.Predicted, C.Labels)

	r, w := m.Probabilities(data)

	if r != nil {
		fmt.Println("Per-class average probabilities for the predicted class in correct predictions of the model")
		fmt.Printf("Av Prob: %3.2f  (stdev %3.2f)\n", r[0], r[1])
	}
	if w == nil {
		return
	}

	fmt.Println("Per-class average probabilities for the predicted class in incorrect predictions of the model")
	fmt.Printf("Av Prob: %3.2f  (stdev %3.2f)\n", w[0], w[1])

}
