package gonet

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"

	"github.com/Lucas-884e/gonet/util"
)

func NewEmbedding(vocabSize, dim int) *Embedding {
	var (
		mat        = make([]*Node, vocabSize*dim)
		normFactor = math.Sqrt(float64(dim))
	)
	for i := range vocabSize {
		for j := range dim {
			v := rand.NormFloat64() / normFactor
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

func EmbeddingLayer(vocabSize, dim int) Layer {
	emb := NewEmbedding(vocabSize, dim)
	return EmbeddingLayerFrom(emb)
}

func EmbeddingLayerFrom(emb *Embedding) Layer {
	return (*embeddingLayer)(emb)
}

type embeddingLayer Embedding

func (el *embeddingLayer) Feed(in []*Node) (out []*Node) {
	return (*Embedding)(el).EmbeddingFeed(in)
}

func (el *embeddingLayer) Parameters() []util.Parameter {
	return util.SliceConvert[*Node, util.Parameter](el.matrix)
}

func (*embeddingLayer) Name() string { return "EmbeddingLayer" }

func DisembeddingLayer(vocabSize, dim int, bias bool) Layer {
	emb := NewEmbedding(vocabSize, dim)
	return DisembeddingLayerFrom(emb, bias)
}

func DisembeddingLayerFrom(emb *Embedding, bias bool) Layer {
	dl := &disembeddingLayer{Embedding: emb}
	if bias {
		dl.bias = make([]*Node, emb.vocabSize)
		for i := range emb.vocabSize {
			dl.bias[i] = NewNode(0, fmt.Sprintf("B_%d", i))
		}
	}
	return dl
}

type disembeddingLayer struct {
	*Embedding
	bias []*Node
}

func (dl *disembeddingLayer) Feed(in []*Node) (out []*Node) {
	if len(in)%dl.dim != 0 {
		panic("input size is not a multiple of disembedding.dim")
	}

	for i := 0; i < len(in); i += dl.dim {
		out = append(out, dl.DisembeddingFeed(in[i:i+dl.dim], dl.bias)...)
	}
	return out
}

func (dl *disembeddingLayer) Parameters() []util.Parameter {
	emb := util.SliceConvert[*Node, util.Parameter](dl.matrix)
	return append(emb, util.SliceConvert[*Node, util.Parameter](dl.bias)...)
}

func (*disembeddingLayer) Name() string { return "DisembeddingLayer" }

func PositionalEmbeddingLayer(vocabSize, ctxLen, dim int) Layer {
	pos := NewInputNodeBatch(ctxLen, "P_%d", false)
	for i, p := range pos {
		p.SetV(float64(i))
	}
	return &positionalEmbeddingLayer{
		semantic:   EmbeddingLayer(vocabSize, dim),
		positional: EmbeddingLayer(ctxLen, dim),
		pos:        pos,
	}
}

type positionalEmbeddingLayer struct {
	semantic   Layer
	positional Layer
	pos        []*Node
}

func (pel *positionalEmbeddingLayer) Feed(in []*Node) (out []*Node) {
	var (
		semOut = pel.semantic.Feed(in)
		posOut = pel.positional.Feed(pel.pos[:len(in)])
	)
	for i, so := range semOut {
		out = append(out, Plus(so, posOut[i]))
	}
	return out
}

func (pel *positionalEmbeddingLayer) Parameters() []util.Parameter {
	return append(pel.semantic.Parameters(), pel.positional.Parameters()...)
}

func (*positionalEmbeddingLayer) Name() string { return "PositionalEmbeddingLayer" }
