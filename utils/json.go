package utils

import (
	"bufio"
	"encoding/json"
	"fmt"
	"sync"
)

type JSONNode struct {
	Id                uint
	Nsamples          int
	Samples           []int
	SplitFeatureIndex int
	BestScoreSoFar    float64
	Leaf              bool
	Threshold         float64
	Leftid            uint
	Rightid           uint
	Branches          int
	Value             float64
	XGB               bool
}

type idGiver struct {
	current uint
	mu      sync.Mutex
}

func (g *idGiver) Next() uint {
	var r uint
	g.mu.Lock()
	r = g.current + 1
	g.current++
	g.mu.Unlock()
	return r

}

type JTree interface {
	JNode(uint) *JSONNode
	//sets the left or right node to the given JTree and return it
	//if given nil, just returns the current value for the node
	Leftf(JTree) JTree
	Rightf(JTree) JTree
	Leaf() bool
}

func UnJSONTree(str string, r *bufio.Reader, creator func(*JSONNode) JTree) (JTree, error) {
	j := new(JSONNode)
	err := json.Unmarshal([]byte(str), j)
	if err != nil {
		return nil, err
	}
	ret := creator(j)
	if j.Leftid > 0 {
		s, err := r.ReadString('\n')
		if err != nil {
			return nil, err
		}
		l, err := UnJSONTree(s, r, creator)
		if err != nil {
			return ret, err
		}
		ret.Leftf(l)
	}
	if j.Rightid > 0 {
		s, err := r.ReadString('\n')
		if err != nil {
			return nil, err
		}

		r, err := UnJSONTree(s, r, creator)
		if err != nil {
			return ret, err
		}
		ret.Rightf(r)

	}
	return ret, nil
}

func JSONTree(t JTree, ids ...*idGiver) ([][]byte, uint, error) {
	var id *idGiver
	if len(ids) == 0 {
		id = &idGiver{} //first tree, the next ID will be one
	} else {
		id = ids[0]
	}
	ID := id.Next()
	var err, errl, errr error
	ret := t.JNode(ID)
	var l, r [][]byte
	if t.Leftf(nil) != nil {
		var leftid uint
		l, leftid, errl = JSONTree(t.Leftf(nil), id)
		ret.Leftid = leftid
	}
	if t.Rightf(nil) != nil {
		var rightid uint
		r, rightid, errr = JSONTree(t.Rightf(nil), id)
		ret.Rightid = rightid
	}
	retstr, err := json.Marshal(ret)
	if err != nil {
		return nil, 0, fmt.Errorf("Error in id: %d, that has %d branches under it: %v", ID, ret.Branches, err)
	}
	rslice := [][]byte{retstr}
	if errl != nil {
		return nil, 0, errl
	}
	if errr != nil {
		return nil, 0, errr
	}
	rslice = append(rslice, l...)
	rslice = append(rslice, r...)
	return rslice, ID, err
}
