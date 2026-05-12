package gonet

import (
	"fmt"
	"strings"
)

func NewNode(v float64, name string) *Node {
	return &Node{
		name:   name,
		v:      v,
		isLeaf: true,
	}
}

func NewInputNode(v float64, name string) *Node {
	n := NewNode(v, name)
	n.isInput = true
	return n
}

func NewInputNodeNoGrad(v float64, name string) *Node {
	n := NewInputNode(v, name)
	n.noGrad = true
	return n
}

func NewInputNodeBatch(size int, nameFmt string, noGrad bool) []*Node {
	batch := make([]*Node, size)
	for i := range batch {
		if noGrad {
			batch[i] = NewInputNodeNoGrad(0, fmt.Sprintf(nameFmt, i))
		} else {
			batch[i] = NewInputNode(0, fmt.Sprintf(nameFmt, i))
		}
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
	noGrad  bool
	prev    []*Node // previous node

	topoOrder []*Node
	forward   func()
	backward  func() // backward propagation function for computing the gradient `g`

	v float64 // value of current node, computed with `forward`
	g float64 // gradient: ∂(next_node)/∂(current_node)
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

func (n *Node) ZeroG() {
	n.g = 0
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
	for _, sn := range n.topologicalSort() {
		if !sn.isLeaf {
			sn.forward()
		}
	}
}

func (n *Node) ForwardBackward() {
	n.Forward()
	n.Backward()
}

func (n *Node) Backward() {
	sorted := n.topologicalSort()
	for _, sn := range sorted {
		sn.g = 0
	}
	n.g = 1

	// `sorted` is in reverse topological order for backward propagation.
	// Therefore, we do `sorted[i].backward()` reversely.
	for i := len(sorted) - 1; i >= 0; i-- {
		if sn := sorted[i]; !sn.isLeaf {
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

func (n *Node) topologicalSort() (sorted []*Node) {
	// We have reused all the graph nodes on every forward and backward propagation,
	// so we don't need to sort every time. Otherwise, it will significantly slow
	// down the training process (by many times).
	if len(n.topoOrder) > 0 {
		return n.topoOrder
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

	n.topoOrder = sorted
	return sorted
}
