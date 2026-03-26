package gonet

import (
	"fmt"
	"math"
	"slices"
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

func NewInputNodeBatch(size int, nameFmt string) []*Node {
	batch := make([]*Node, size)
	for i := range batch {
		batch[i] = NewInputNode(0, fmt.Sprintf(nameFmt, i+1))
	}
	return batch
}

func NodeValues(ns []*Node) []float64 {
	vs := make([]float64, len(ns))
	for i, n := range ns {
		vs[i] = n.v
	}
	return vs
}

type Node struct {
	name    string
	isLeaf  bool // Is current node a leaf node (ie, not a composite note generated from a few other nodes)
	isInput bool // Is current node an input node (input values for neural network input layer)?
	op      Operator
	prev    []*Node // previous node

	v            float64 // value of current node, computed with `forward`
	forward      func()
	forwardOrder []*Node

	g             float64 // gradient: ∂(next_node)/∂(current_node)
	backward      func()  // backward propagation function for computing the gradient `g`
	backwardOrder []*Node
}

func (n *Node) SetName(name string) {
	n.name = name
}

func (n *Node) Name() string {
	return n.name
}

func (n *Node) SetV(v float64) {
	n.v = v
}

func (n *Node) V() float64 {
	return n.v
}

func (n *Node) G() float64 {
	return n.g
}

func (n *Node) Learn(delta float64) {
	n.v -= delta
}

func (n *Node) Forward() {
	// We do breadth-first-search first so that we don't do forward propagation
	// on the same node more than once.
	for _, node := range n.bfs() {
		node.forward()
	}
}

func (n *Node) Backward() {
	n.Forward()

	sorted := n.topologicalSort()
	for _, sn := range sorted {
		sn.g = 0
	}
	n.g = 1

	for _, sn := range sorted {
		if !sn.isLeaf {
			sn.backward()
		}
	}
}

func (n *Node) String() string {
	var (
		sorted = n.topologicalSort()
		sb     = new(strings.Builder)
	)
	sb.WriteByte('\n')
	for _, sn := range sorted {
		sb.WriteString(sn.name)
		fmt.Fprintf(sb, " | value=%.6g", sn.v)
		fmt.Fprintf(sb, " | gradient=%.6g", sn.g)
		sb.WriteByte('\n')
	}
	return sb.String()
}

func (n *Node) bfs() (nodes []*Node) {
	// We have reused all the graph nodes on every forward and backward propagation,
	// so we don't need to traversal every time. Otherwise, it will significantly
	// slow down the training process (by many times).
	if len(n.forwardOrder) > 0 {
		return n.forwardOrder
	}

	var (
		visited = make(map[*Node]bool)
		stack   = []*Node{n}
	)
	for len(stack) > 0 {
		pop := stack[0]
		stack = stack[1:]
		if !pop.isLeaf {
			nodes = append(nodes, pop)
		}

		// Traverse in reverse order so that after `slices.Reverse`, the result
		// sequence will be in natural order. Otherwise, the forward propagation
		// of some types (eg, Softmax) of Nodes would fail because they depends on
		// the first node being forward propagated first.
		for i := len(pop.prev) - 1; i >= 0; i-- {
			if p := pop.prev[i]; !visited[p] && !p.isLeaf {
				visited[p] = true
				stack = append(stack, p)
			}
		}
	}

	slices.Reverse(nodes)
	n.forwardOrder = nodes
	return nodes
}

func (n *Node) topologicalSort() (sorted []*Node) {
	// We have reused all the graph nodes on every forward and backward propagation,
	// so we don't need to sort every time. Otherwise, it will significantly slow
	// down the training process (by many times).
	if len(n.backwardOrder) > 0 {
		return n.backwardOrder
	}

	var (
		sort    func(*Node)
		visited = make(map[*Node]bool)
	)
	sort = func(curr *Node) {
		if !visited[curr] {
			visited[curr] = true
			for _, p := range curr.prev {
				sort(p)
			}

			// Append current node after sorting previous nodes. This is especially
			// important for recurrent neural networks where one previous node might
			// depend on another previous node of the same current node.
			// NOTE: although we don't do backward propagation on leaf nodes
			// (ie, weight nodes), we still need to put them in the sorted list as we
			// need to keep track all the nodes whose gradients have to be cleared
			// before the training on each new batch.
			sorted = append(sorted, curr)
		}
	}
	sort(n)

	// We appended current node after sorting previous nodes, so the `sorted` is
	// in reverse topological order. Therefore, we do `Reverse` so that we can do
	// back-propagation in normal order.
	slices.Reverse(sorted)
	n.backwardOrder = sorted
	return sorted
}

func Plus(prev ...*Node) *Node {
	if len(prev) < 2 {
		panic("+ node must have at least two previous nodes")
	}

	var names []string
	for _, n := range prev {
		names = append(names, n.name)
	}
	out := &Node{
		name: strings.Join(names, "+"),
		op:   OpPlus,
		prev: prev,
	}
	out.forward = func() {
		out.v = 0
		for _, n := range prev {
			out.v += n.v
		}
	}
	out.backward = func() {
		for _, n := range prev {
			if !n.isInput {
				n.g += out.g
			}
		}
	}
	return out
}

func Multiply(prev ...*Node) *Node {
	if len(prev) < 2 {
		panic("× node must have at least two previous nodes")
	}

	var (
		names  []string
		localG = make([]float64, len(prev))
	)
	for _, n := range prev {
		if n.op == OpPlus {
			names = append(names, "("+n.name+")")
		} else {
			names = append(names, n.name)
		}
	}
	out := &Node{
		name: strings.Join(names, "×"),
		op:   OpMultiply,
		prev: prev,
	}
	out.forward = func() {
		out.v = 1
		for i, n := range prev {
			for j := 0; j <= i; j++ {
				if j == i {
					localG[j] = out.v
				} else {
					localG[j] *= n.v
				}
			}
			out.v *= n.v
		}
	}
	out.backward = func() {
		for i, n := range prev {
			if !n.isInput {
				n.g += localG[i] * out.g
			}
		}
	}
	return out
}

func Relu(prev *Node) *Node {
	out := &Node{
		name: fmt.Sprintf("relu(%s)", prev.name),
		op:   OpRelu,
		prev: []*Node{prev},
	}
	out.forward = func() {
		out.v = 0
		if prev.v > 0 {
			out.v = prev.v
		}
	}
	out.backward = func() {
		if out.v > 0 {
			prev.g += out.g
		}
	}
	return out
}

func Sigmoid(prev *Node) *Node {
	out := &Node{
		name: fmt.Sprintf("σ(%s)", prev.name),
		op:   OpSigmoid,
		prev: []*Node{prev},
	}
	out.forward = func() {
		out.v = 1 / (1 + math.Exp(-prev.v))
	}
	out.backward = func() {
		prev.g += out.v * (1 - out.v) * out.g
	}
	return out
}

func Tanh(prev *Node) *Node {
	out := &Node{
		name: fmt.Sprintf("tanh(%s)", prev.name),
		op:   OpTanh,
		prev: []*Node{prev},
	}
	out.forward = func() {
		out.v = math.Tanh(prev.v)
	}
	out.backward = func() {
		prev.g += (1 - out.v) * (1 + out.v) * out.g
	}
	return out
}

// Softmax also accepts a parameter `t`, which is sometimes called temperature.
func Softmax(t float64, prev ...*Node) []*Node {
	if len(prev) < 2 {
		panic("softmax node must have at least two previous nodes")
	}

	outs := make([]*Node, len(prev))
	for i := range prev {
		outs[i] = &Node{
			name: fmt.Sprintf("softmax[index=%d](T=%g)", i, t),
			op:   OpSoftmax,
			prev: prev,
		}
	}

	var (
		sum float64
		ys  = make([]float64, len(prev))
	)
	for i, out := range outs {
		// NOTE: this must be consistent with the traversal order of forward
		// propagation of the network. If it's not the first (i==0) output forward
		// running first (ie, `sum` is not computed first), it would result in
		// completely wrong `out.v`.
		if i == 0 {
			out.forward = func() {
				sum = 0
				for j, n := range prev {
					ys[j] = math.Exp(n.v / t)
					sum += ys[j]
				}
				out.v = ys[0] / sum
			}
		} else {
			// NOTE: This closure only works for Go1.22+ because `i` doesn't preserve
			// the value on the out.forward assignment before this version.
			out.forward = func() {
				out.v = ys[i] / sum
			}
		}

		// NOTE: This closure only works for Go1.22+ because `i` doesn't preserve
		// the value on the out.forward assignment before this version.
		out.backward = func() {
			for j, n := range prev {
				if j == i {
					n.g += out.v * (1 - out.v) * out.g
				} else {
					n.g -= out.v * outs[j].v * out.g
				}
			}
		}
	}

	return outs
}
