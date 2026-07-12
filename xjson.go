package boo

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"strings"

	"github.com/rmera/boo/utils"
)

type writestringer interface {
	WriteString(string) (int, error)
}

type jsonTester struct {
	Str []string
}

func newjsonTester() *jsonTester {
	j := new(jsonTester)
	j.Str = make([]string, 0, 10)
	return j
}
func (j *jsonTester) WriteString(w string) (int, error) {
	j.Str = append(j.Str, w)
	return 0, nil
}

var ActivationMap map[string]utils.Activation = map[string]utils.Activation{
	"softmax":       &utils.SoftMax{},
	"identity":      &utils.Identity{},
	"normalization": &utils.Normalization{},
}

// LossMap maps the name returned by a utils.LossFunc's Name method to
// an instance of that loss function. It's used to recover the Loss
// field of an Options struct from the name saved in a JSON file.
var LossMap map[string]utils.LossFunc = map[string]utils.LossFunc{
	"sqerr": &utils.SQErrLoss{},
	"mse":   &utils.MSELoss{},
}

func UnJSONMultiClass(r *bufio.Reader, opts ...*Options) (*MultiClass, error) {
	ret := &MultiClass{}
	jmc := &JSONMetaData{}
	s, err := r.ReadString('\n')
	if err != nil {
		return nil, fmt.Errorf("Error reading metadata from file: %v", err)
	}
	err = json.Unmarshal([]byte(s), jmc)
	if err != nil {
		return nil, fmt.Errorf("Error unmarshalling metadata: %v", err)
	}
	ret.learningRate = jmc.LearningRate
	ret.classLabels = jmc.ClassLabels
	ret.activation = ActivationMap[jmc.ActiName]
	ret.baseScore = jmc.BaseScore
	if len(opts) > 0 && opts[0] != nil && jmc.Options != nil {
		*opts[0] = *jmc.Options.toOptions()
	}
	//I'm not sure this will work!
	//	s, err = r.ReadString('\n')
	//	if err != nil {
	//		return nil, fmt.Errorf("Error reading trees from file: %v", err)
	//	}
	trees := make([][]*Tree, 0, 2)
	var class []*Tree
	cont := 1
	nround := -1
	nclass := 0
	for {
		s, err = r.ReadString('\n')
		if err != nil {
			break
		}
		if strings.Contains(s, "ROUND") {
			if class != nil {
				trees = append(trees, class)
			}
			nround++
			nclass = 0
			class = make([]*Tree, 0, 1)
			continue
		}

		if strings.Contains(s, "CLASS") {
			continue
		}
		jtree, err := utils.UnJSONTree(s, r, creator)
		if err != nil {
			return nil, fmt.Errorf("Error reading tree %d round %d, class %d: %v", cont, nround, nclass, err)
		}
		class = append(class, jtree.(*Tree))
		nclass++
		cont++
	}
	if err.Error() != "EOF" {
		return nil, fmt.Errorf("Error reading of trees lines from file: %v", err)

	}
	ret.b = trees
	return ret, nil
}

// Marshals a multi-class classifier to JSON. probtransformname is the name of the activation
// function, normally, "softmax", w is any object with a WriteString(string)(int,error)
// method, normally, a *bufio.Writer. An optional *Options can be given, in which case its
// values are saved as part of the file's metadata. If no *Options is given, the produced
// file is identical to what the previous version of this function would have produced, and
// remains readable by it.
func JSONMultiClass(m *MultiClass, activationfunctionname string, w writestringer, opts ...*Options) error {
	j, err := MarshalMCMetaData(m, activationfunctionname, opts...)
	if err != nil {
		return err
	}
	_, err = w.WriteString(string(j))
	if err != nil {
		return err
	}
	for rn, round := range m.b {
		_, err = w.WriteString(fmt.Sprintf("ROUND %d\n", rn))
		if err != nil {
			return err
		}
		for cn, class := range round {
			_, err = w.WriteString(fmt.Sprintf("CLASS %d, label: %d \n", cn, m.classLabels[cn]))
			if err != nil {
				return err
			}
			tree, _, err := utils.JSONTree(class)
			if err != nil {
				return err
			}
			_, err = w.WriteString(string(bytes.Join(tree, []byte("\n"))) + "\n")
			if err != nil {
				return err
			}
		}
	}
	return nil
}

type JSONMetaData struct {
	LearningRate float64
	ClassLabels  []int
	ActiName     string
	BaseScore    float64
	// Options carries the hyperparameters used to train the model. It's
	// only present if a *Options was given to MarshalMCMetaData/JSONMultiClass,
	// so its absence (nil) doesn't break unmarshalling of files produced
	// before this field existed.
	Options *JSONOptions `json:",omitempty"`
}

// JSONOptions mirrors Options, replacing the non-serializable Loss
// field (a utils.LossFunc interface) with the name returned by its
// Name method, so it can be round-tripped through JSON. See LossMap.
type JSONOptions struct {
	XGB            bool
	Rounds         int
	MaxDepth       int
	EarlyStop      int
	LearningRate   float64
	Lambda         float64
	MinChildWeight float64
	Gamma          float64
	SubSample      float64
	ColSubSample   float64
	BaseScore      float64
	Regression     bool
	//	MinSample      int
	TreeMethod string
	Verbose    bool
	LossName   string
	ActiName   string
}

// optionsToJSONOptions converts an *Options into its JSON-friendly
// representation. Returns nil if o is nil.
func optionsToJSONOptions(o *Options) *JSONOptions {
	if o == nil {
		return nil
	}
	lossname := ""
	if o.Loss != nil {
		lossname = o.Loss.Name()
	}
	actiname := ""
	if o.Activation != nil {
		actiname = o.Activation.Name()
	}
	return &JSONOptions{
		XGB:            o.XGB,
		Rounds:         o.Rounds,
		MaxDepth:       o.MaxDepth,
		EarlyStop:      o.EarlyStop,
		LearningRate:   o.LearningRate,
		Lambda:         o.Lambda,
		MinChildWeight: o.MinChildWeight,
		Gamma:          o.Gamma,
		SubSample:      o.SubSample,
		ColSubSample:   o.ColSubSample,
		BaseScore:      o.BaseScore,
		Regression:     o.Regression(),
		//MinSample:      o.MinSample,
		TreeMethod: o.TreeMethod,
		Verbose:    o.Verbose,
		LossName:   lossname,
		ActiName:   actiname,
	}
}

// toOptions converts a JSONOptions back into an *Options, recovering
// the Loss field from LossMap. Returns nil if jo is nil.
func (jo *JSONOptions) toOptions() *Options {
	if jo == nil {
		return nil
	}
	ret := &Options{
		XGB:            jo.XGB,
		Rounds:         jo.Rounds,
		MaxDepth:       jo.MaxDepth,
		EarlyStop:      jo.EarlyStop,
		LearningRate:   jo.LearningRate,
		Lambda:         jo.Lambda,
		MinChildWeight: jo.MinChildWeight,
		Gamma:          jo.Gamma,
		SubSample:      jo.SubSample,
		ColSubSample:   jo.ColSubSample,
		BaseScore:      jo.BaseScore,
		MinSample:      0, //jo.MinSample,
		TreeMethod:     jo.TreeMethod,
		Verbose:        jo.Verbose,
		Loss:           LossMap[jo.LossName],
		Activation:     ActivationMap[jo.ActiName],
	}

	ret.Regression(jo.Regression)
	return ret
}

// MarshalMCMetaData marshals the metadata for a MultiClass model. An
// optional *Options can be given, in which case it's included in the
// resulting JSON (see JSONMetaData.Options). If none is given, the
// output is identical to what this function produced before Options
// support was added, and stays readable by UnJSONMultiClass without
// giving it an *Options.
func MarshalMCMetaData(m *MultiClass, probtransformname string, opts ...*Options) ([]byte, error) {
	r := &JSONMetaData{
		LearningRate: m.learningRate,
		ClassLabels:  m.classLabels,
		ActiName:     m.activation.Name(),
		BaseScore:    m.baseScore,
	}
	if len(opts) > 0 {
		r.Options = optionsToJSONOptions(opts[0])
	}
	j, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	j = append(j, '\n')
	return j, nil
}

func (t *Tree) JNode(id uint, addsamples ...bool) *utils.JSONNode {
	bs := t.bestScoreSoFar
	if t.Leaf() && !t.xgb {
		bs = 0.1189998819991197253
	}
	ret := &utils.JSONNode{
		Id:                id,
		Nsamples:          t.nsamples,
		Leaf:              t.Leaf(),
		Threshold:         t.threshold,
		XGB:               t.xgb,
		Branches:          t.branches,
		BestScoreSoFar:    bs,
		SplitFeatureIndex: t.splitFeatureIndex,
		Value:             t.value,
		Leftid:            0,
		Rightid:           0,
	}
	if len(addsamples) > 0 && addsamples[0] {
		ret.Samples = t.samples
	}
	return ret
}

func (T *Tree) Leftf(l utils.JTree) utils.JTree {
	if l != nil {
		T.left = l.(*Tree)
	}

	if T.left == nil {
		return nil
	}
	return T.left
}

func (T *Tree) Rightf(r utils.JTree) utils.JTree {
	if r != nil {
		T.right = r.(*Tree)
	}
	if T.right == nil {
		return nil
	}
	return T.right
}

func creator(j *utils.JSONNode) utils.JTree {
	ret := &Tree{
		bestScoreSoFar:    j.BestScoreSoFar,
		value:             j.Value,
		samples:           j.Samples,
		nsamples:          j.Nsamples,
		splitFeatureIndex: j.SplitFeatureIndex,
		threshold:         j.Threshold,
		branches:          j.Branches,
		xgb:               j.XGB,
	}
	if j.Leaf && !j.XGB {
		ret.bestScoreSoFar = math.Inf(0)
	}
	return ret
}
