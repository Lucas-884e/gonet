package gonet

import (
	"fmt"
	"math/rand/v2"
	"strings"
)

func NewEmbedding(vocabSize, dim int) *Embedding {
	mat := make([]*Node, vocabSize*dim)
	for i := range vocabSize {
		for j := range dim {
			v := rand.NormFloat64()
			mat[i*dim+j] = NewNode(v, fmt.Sprintf("E_%d_%d", j, i))
		}
	}
	return &Embedding{
		matrix:    mat,
		vocabSize: vocabSize,
		dim:       dim,
	}
}

type Embedding struct {
	matrix    []*Node
	vocabSize int
	dim       int
}

func (e *Embedding) E(index int) []*Node {
	start := index * e.dim
	return e.matrix[start : start+e.dim]
}

func (e *Embedding) component(eindex, cindex int) *Node {
	return e.matrix[eindex*e.dim+cindex]
}

func (e *Embedding) EmbeddingFeed(in []*Node) (out []*Node) {
	noGrad := in[0].noGrad
	for _, n := range in {
		for cidx := range e.dim {
			o := &Node{
				name:   fmt.Sprintf("Embedding_%d", cidx),
				noGrad: noGrad,
			}
			o.forward = func() {
				eidx := int(n.V())
				o.v = e.component(eidx, cidx).v
			}
			if !noGrad {
				o.backward = func() {
					eidx := int(n.V())
					w := e.component(eidx, cidx)
					w.g += o.g
				}
			}
			out = append(out, o)
		}
	}
	return out
}

func (e *Embedding) DisembeddingFeed(in, bias []*Node) (out []*Node) {
	if len(in) != e.dim {
		panic(fmt.Sprintf("Feed input size %d does not match embedding dimension %d", len(in), e.dim))
	}

	withBias := len(bias) > 0
	for i := range e.vocabSize {
		// Preserve one slot capacity for bias node, if any.
		prod := make([]*Node, e.dim, e.dim+1)
		for j, x := range in {
			prod[j] = Multiply(e.matrix[i*e.dim+j], x)
		}
		if withBias {
			prod = append(prod, bias[i])
		}
		out = append(out, Plus(prod...))
	}
	return out
}

func (e *Embedding) Sub(i, j int) (diff []float64) {
	ei := e.E(i)
	ej := e.E(j)
	for k, v := range ei {
		diff = append(diff, v.V()-ej[k].V())
	}
	return diff
}

func (e *Embedding) S(index int) string {
	emb := e.E(index)
	sb := new(strings.Builder)
	sb.WriteByte('[')
	for i, v := range emb {
		if i > 0 {
			sb.WriteString(", ")
		}
		fmt.Fprintf(sb, "%.6g", v.V())
	}
	sb.WriteByte(']')
	return sb.String()
}
