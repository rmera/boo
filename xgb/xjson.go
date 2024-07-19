package xgb

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"strings"

	"github.com/rmera/learn"
	"gonum.org/v1/gonum/mat"
)

type writestringer interface {
	WriteString(string) (int, error)
}

var ProbTransformMap map[string]func(*mat.Dense, *mat.Dense) *mat.Dense = map[string]func(*mat.Dense, *mat.Dense) *mat.Dense{
	"softmax": learn.SoftMaxDense,
}

func UnJSONMultiClass(r *bufio.Reader) (*MultiClass, error) {
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
	ret.probTransform = ProbTransformMap[jmc.ProbTransformName]
	ret.baseScore = jmc.BaseScore
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
		jtree, err := learn.UnJSONTree(s, r, creator)
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

func JSONMultiClass(m *MultiClass, probtransformname string, w writestringer) error {
	j, err := MarshalMCMetaData(m, probtransformname)
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
			tree, _, err := learn.JSONTree(class)
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
	LearningRate      float64
	ClassLabels       []int
	ProbTransformName string
	BaseScore         float64
}

func MarshalMCMetaData(m *MultiClass, probtransformname string) ([]byte, error) {
	r := &JSONMetaData{
		LearningRate:      m.learningRate,
		ClassLabels:       m.classLabels,
		ProbTransformName: probtransformname,
		BaseScore:         m.baseScore,
	}
	j, err := json.Marshal(r)
	if err != nil {
		return nil, err
	}
	j = append(j, '\n')
	return j, nil
}

func (t *Tree) JNode(id uint) *learn.JSONNode {
	ret := &learn.JSONNode{
		Id:                id,
		Samples:           t.samples,
		Nsamples:          t.nsamples,
		Leaf:              t.Leaf(),
		Threshold:         t.threshold,
		Branches:          t.branches,
		BestScoreSoFar:    t.bestScoreSoFar,
		SplitFeatureIndex: t.splitFeatureIndex,
		Value:             t.value,
		Leftid:            0,
		Rightid:           0,
	}
	return ret
}

func (T *Tree) Leftf(l learn.JTree) learn.JTree {
	if l != nil {
		T.Left = l.(*Tree)
	}

	if T.Left == nil {
		return nil
	}
	return T.Left
}

func (T *Tree) Rightf(r learn.JTree) learn.JTree {
	if r != nil {
		T.Right = r.(*Tree)
	}
	if T.Right == nil {
		return nil
	}
	return T.Right
}

func creator(j *learn.JSONNode) learn.JTree {
	ret := &Tree{
		bestScoreSoFar:    j.BestScoreSoFar,
		value:             j.Value,
		samples:           j.Samples,
		nsamples:          j.Nsamples,
		splitFeatureIndex: j.SplitFeatureIndex,
		threshold:         j.Threshold,
		branches:          j.Branches,
	}
	return ret
}
