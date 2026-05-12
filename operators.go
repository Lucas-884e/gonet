package gonet

import (
	"cmp"
	"fmt"
	"math"
	"slices"
	"strings"
)

func Identity(n *Node) *Node { return n }

func Plus(prev ...*Node) *Node {
	if len(prev) < 2 {
		panic("+ node must have at least two previous nodes")
	}

	var names []string
	for _, n := range prev {
		names = append(names, n.name)
	}
	var (
		noGrad = prev[0].noGrad
		out    = &Node{
			name:   strings.Join(names, "+"),
			prev:   prev,
			noGrad: noGrad,
		}
	)
	out.forward = func() {
		out.v = 0
		for _, n := range prev {
			out.v += n.v
		}
	}
	if !noGrad {
		out.backward = func() {
			for _, n := range prev {
				if !n.isInput {
					n.g += out.g
				}
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
		noGrad bool
	)
	for _, n := range prev {
		if n.isLeaf {
			names = append(names, n.name)
		} else {
			names = append(names, "("+n.name+")")
		}
		if n.noGrad {
			noGrad = true
		}
	}
	out := &Node{
		name:   strings.Join(names, "×"),
		prev:   prev,
		noGrad: noGrad,
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
	if !noGrad {
		out.backward = func() {
			for i, n := range prev {
				if !n.isInput {
					n.g += localG[i] * out.g
				}
			}
		}
	}
	return out
}

func Relu(prev *Node) *Node {
	out := &Node{
		name:   fmt.Sprintf("relu(%s)", prev.name),
		prev:   []*Node{prev},
		noGrad: prev.noGrad,
	}
	out.forward = func() {
		out.v = 0
		if prev.v > 0 {
			out.v = prev.v
		}
	}
	if !prev.noGrad {
		out.backward = func() {
			if out.v > 0 {
				prev.g += out.g
			}
		}
	}
	return out
}

func Sigmoid(prev *Node) *Node {
	out := &Node{
		name:   fmt.Sprintf("σ(%s)", prev.name),
		prev:   []*Node{prev},
		noGrad: prev.noGrad,
	}
	out.forward = func() {
		out.v = 1 / (1 + math.Exp(-prev.v))
	}
	if !prev.noGrad {
		out.backward = func() {
			prev.g += out.v * (1 - out.v) * out.g
		}
	}
	return out
}

func Tanh(prev *Node) *Node {
	out := &Node{
		name:   fmt.Sprintf("tanh(%s)", prev.name),
		prev:   []*Node{prev},
		noGrad: prev.noGrad,
	}
	out.forward = func() {
		out.v = math.Tanh(prev.v)
	}
	if !prev.noGrad {
		out.backward = func() {
			prev.g += (1 - out.v) * (1 + out.v) * out.g
		}
	}
	return out
}

// Softmax also accepts a parameter `t`, which is sometimes called temperature.
func Softmax(t float64, prev ...*Node) []*Node {
	if len(prev) < 2 {
		panic("softmax node must have at least two previous nodes")
	}

	var (
		outs   = make([]*Node, len(prev))
		noGrad = prev[0].noGrad
	)
	for i := range prev {
		outs[i] = &Node{
			name:   fmt.Sprintf("softmax[index=%d](T=%g)", i, t),
			prev:   prev,
			noGrad: noGrad,
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
				vmax := slices.MaxFunc(prev, func(a, b *Node) int { return cmp.Compare(a.v, b.v) }).v
				sum = 0
				for j, n := range prev {
					ys[j] = math.Exp((n.v - vmax) / t)
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

		if !noGrad {
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
	}

	return outs
}

func Linear(ws, xs []*Node, bias *Node, names ...string) *Node {
	var (
		noGrad = cmp.Or(ws[0].noGrad, xs[0].noGrad)
		out    = &Node{
			name:   "Linear",
			prev:   append(append([]*Node{}, ws...), xs...),
			noGrad: noGrad,
		}
	)
	if bias != nil {
		out.prev = append(out.prev, bias)
	}
	if len(names) > 0 {
		out.name = names[0]
	}

	out.forward = func() {
		out.v = 0
		for i, x := range xs {
			out.v += ws[i].v * x.v
		}
		if bias != nil {
			out.v += bias.v
		}
	}

	if !noGrad {
		out.backward = func() {
			for i, w := range ws {
				w.g += xs[i].v * out.g
				xs[i].g += w.v * out.g
			}
			if bias != nil {
				bias.g += out.g
			}
		}
	}

	return out
}

func DotProduct(left, right []*Node) *Node {
	return Linear(left, right, nil, "DotProduct")
}

func VectorAdd(left, right []*Node) (out []*Node) {
	for i, elem := range right {
		out = append(out, Plus(left[i], elem))
	}
	return out
}
