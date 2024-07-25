package utils

import (
	"bufio"
	"encoding/json"
	"errors"
	"fmt"
	"sync"
)

// Structure for serializing a tree node.
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

func (j *JSONNode) String() string {
	r := fmt.Sprintf("ID:%d, Samples:%v, ns: %d, score: %.3f,Value:%.3f", j.Id, j.Samples, j.Nsamples, j.BestScoreSoFar, j.Value)
	r2 := fmt.Sprintf("Threshold: %.3f, Leaf: %v, Branches: %d, LeftID: %d, ", j.Threshold, j.Leaf, j.Branches, j.Leftid)
	r3 := fmt.Sprintf("RightID: %d, XGB:%v", j.Rightid, j.XGB)
	return r + r2 + r3
}

// This returns a new and consecutive ID every time it's Next() metohd is called, starting with 1. It is concurrent-safe.
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

// A tree that can "pack" itself as a JSONNode
type JTree interface {
	JNode(uint) *JSONNode
	//sets the left or right node to the given JTree and return it
	//if given nil, just returns the current value for the node
	Leftf(JTree) JTree
	Rightf(JTree) JTree
	Leaf() bool
}

// takes a bufio reader containing a json-serialized representation of a Tree, plus a
// tree creator function, and returns the created JTree
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

// takes a JSON-able tree and returns it as a slice of json-serialized []byte,
// where each element in the slice is a node.
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
		return nil, 0, errors.Join(fmt.Errorf("Error in id: %d, that has %d branches under it: %v", ID, ret.Branches, ret), err)
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
