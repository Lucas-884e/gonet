package graph

import (
	"fmt"
	"math"
	"strings"
)

func NewNode(v float64, name string) *Node {
	return &Node{
		name:   name,
		v:      v,
		isLeaf: true,
		op:     OpNone,
	}
}

func NewInputNode(v float64, name string) *Node {
	n := NewNode(v, name)
	n.isInput = true
	return n
}

type Node struct {
	name     string
	v        float64 // value of current node
	g        float64 // gradient: ∂(next_node)/∂(current_node)
	isLeaf   bool    // Is current node a leaf node (ie, not a composite note generated from a few other nodes)
	isInput  bool    // Is current node an input node (input values for neural network input layer)?
	op       Operator
	prev     []*Node // previous node
	backward func()  // backward propagation function for computing the gradient `g`
}

func (n *Node) Name() string {
	return n.name
}

func (n *Node) V() float64 {
	return n.v
}

func (n *Node) G() float64 {
	return n.g
}

func (n *Node) Learn(rate float64) {
	n.v -= rate * n.g
}

func (n *Node) Backward() {
	if n.isInput {
		return
	}

	var (
		sort    func(*Node)
		sorted  []*Node
		visited = make(map[*Node]bool)
	)
	sort = func(curr *Node) {
		if !visited[curr] {
			visited[curr] = true
			for _, p := range curr.prev {
				sort(p)
			}
			// Append current node after sorting previous nodes. This is especially
			// imported for recurrent neural networks where one previous node might
			// depend on another previous node of the same current node.
			sorted = append(sorted, curr)
		}
	}
	sort(n)

	n.g = 1
	// We appended current node after sorting previous nodes, so the `sorted` is
	// in reverse topological order. Therefore, we should call their backward
	// function in reverse order.
	for i := len(sorted) - 1; i >= 0; i-- {
		if sn := sorted[i]; !sn.isLeaf {
			sn.backward()
		}
	}
}

func Plus(ns ...*Node) *Node {
	if len(ns) < 2 {
		panic("+ node must have at least two previous nodes")
	}

	var (
		names []string
		v     float64
	)
	for _, n := range ns {
		names = append(names, n.name)
		v += n.v
	}
	out := &Node{
		name: strings.Join(names, "+"),
		v:    v,
		op:   OpPlus,
		prev: ns,
	}
	out.backward = func() {
		for _, n := range ns {
			if !n.isInput {
				n.g += out.g
			}
		}
	}
	return out
}

func Multiply(ns ...*Node) *Node {
	if len(ns) < 2 {
		panic("× node must have at least two previous nodes")
	}

	var (
		names  []string
		v      = 1.0
		localG = make([]float64, len(ns))
	)
	for i, n := range ns {
		names = append(names, n.name)
		for j := 0; j <= i; j++ {
			if j == i {
				localG[j] = v
			} else {
				localG[j] *= n.v
			}
		}
		v *= n.v
	}
	out := &Node{
		name: strings.Join(names, "×"),
		v:    v,
		op:   OpMultiply,
		prev: ns,
	}
	out.backward = func() {
		for i, n := range ns {
			if !n.isInput {
				n.g += localG[i] * out.g
			}
		}
	}
	return out
}

func Relu(n *Node) *Node {
	var v float64
	if n.v > 0 {
		v = n.v
	}
	out := &Node{
		name: fmt.Sprintf("relu(%s)", n.name),
		v:    v,
		op:   OpRelu,
		prev: []*Node{n},
	}
	out.backward = func() {
		if v > 0 {
			n.g += out.g
		}
	}
	return out
}

func Sigmoid(n *Node) *Node {
	out := &Node{
		name: fmt.Sprintf("σ(%s)", n.name),
		v:    1 / (1 + math.Exp(-n.v)),
		op:   OpSigmoid,
		prev: []*Node{n},
	}
	out.backward = func() {
		n.g += out.v * (1 - out.v) * out.g
	}
	return out
}

func Tanh(n *Node) *Node {
	out := &Node{
		name: fmt.Sprintf("tanh(%s)", n.name),
		v:    math.Tanh(n.v),
		op:   OpTanh,
		prev: []*Node{n},
	}
	out.backward = func() {
		n.g += (1 - out.v) * (1 + out.v) * out.g
	}
	return out
}

// Softmax also accepts a parameter `t`, which is sometimes called temperature.
func Softmax(t float64, ns ...*Node) (outs []*Node) {
	if len(ns) < 2 {
		panic("softmax node must have at least two previous nodes")
	}

	var sum float64
	ys := make([]float64, len(ns))
	for i, n := range ns {
		ys[i] = math.Exp(n.v / t)
		sum += ys[i]
	}
	for i, y := range ys {
		ys[i] = y / sum
	}

	for i, y := range ys {
		outs = append(outs, &Node{
			name: fmt.Sprintf("softmax[index=%d](T=%g)", i, t),
			v:    y,
			op:   OpSoftmax,
			prev: ns,
		})
	}
	for j, out := range outs {
		out.backward = func() {
			for i, n := range ns {
				if i == j {
					n.g += out.v * (1 - out.v) * out.g
				} else {
					n.g -= out.v * outs[i].v * out.g
				}
			}
		}
	}
	return
}
