package confu

import (
	"fmt"
	"testing"

	"github.com/rmera/boo"
	"github.com/rmera/boo/utils"
)

func buildConf(Te *testing.T) (*Confusions, map[int]string) {
	data, err := utils.DataBunchFromLibSVMFile("../tests/train.svm", true)
	if err != nil {
		Te.Error(err)
	}
	datat, err := utils.DataBunchFromLibSVMFile("../tests/test.svm", true)
	if err != nil {
		Te.Error(err)
	}

	O := boo.DefaultXOptions()
	O.Rounds = 100
	O.EarlyStop = 10
	O.Verbose = false

	boosted := boo.NewMultiClass(data, O)
	fmt.Println("test set accuracy", boosted.Accuracy(datat))
	namemap, err := ReadNameMap("../tests/labeldic.dat")
	if err != nil {
		Te.Error(err)
	}
	return MCConfusions(boosted, datat), namemap

}

func TTestTopNClass3(Te *testing.T) {
	fmt.Println("TestTopN Seps")
	conf, _ := buildConf(Te)
	for i := 1; i < 4; i++ {
		r, rf := conf.TopNForLabelM(3, i)
		fmt.Println(r, rf)
	}

}

func TestTopN(Te *testing.T) {
	fmt.Println("TestTopN")
	conf, _ := buildConf(Te)
	r, rf := conf.TopNPerLabel(3)
	for i, v := range r {
		fmt.Println(v, rf[i])
	}

}

func TestMatrix(Te *testing.T) {
	fmt.Println("TestMatrix")
	conf, _ := buildConf(Te)
	mat := conf.Matrix()
	fmt.Println(conf.Labels, conf.m)
	for _, v := range mat {
		fmt.Println(v)
	}
}

func TestConfu(Te *testing.T) {
	conf, namemap := buildConf(Te)
	topn := conf.PrintTopN(5, namemap)
	fmt.Printf(topn + "\n")

}
