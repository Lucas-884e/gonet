package gonet

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"

	"github.com/LucasInOz/gonet/util"
)

func NewEmbedding(vocabSize, dim int, initNorm bool, names ...string) *Embedding {
	var (
		mat     = make([]*Node, vocabSize*dim)
		divisor = 1.0
		name    = "Embedding"
	)
	if initNorm {
		divisor = math.Sqrt(float64(dim))
	}
	if len(names) > 0 {
		name = names[0]
	}
	for i := range vocabSize {
		for j := range dim {
			v := rand.NormFloat64() / divisor
			mat[i*dim+j] = NewNode(v, fmt.Sprintf("%s_%d_%d", name, j, i))
		}
	}
	return &Embedding{
		matrix:    mat,
		vocabSize: vocabSize,
		dim:       dim,
		name:      name,
	}
}

type Embedding struct {
	matrix    []*Node
	vocabSize int
	dim       int
	name      string
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
				name:   fmt.Sprintf("%s_%d", e.name, cidx),
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

func (e *Embedding) UnembeddingFeed(in, bias []*Node) (out []*Node) {
	if len(in) != e.dim {
		panic(fmt.Sprintf("Feed input size %d does not match unembedding dimension %d", len(in), e.dim))
	}

	for i := range e.vocabSize {
		weights := e.matrix[i*e.dim : (i+1)*e.dim]
		if len(bias) > 0 {
			out = append(out, Linear(weights, in, bias[i]))
		} else {
			out = append(out, Linear(weights, in, nil))
		}
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

func EmbeddingLayer(vocabSize, dim int, names ...string) Layer {
	emb := NewEmbedding(vocabSize, dim, false, names...)
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

func UnembeddingLayer(dim, vocabSize int, bias bool) Layer {
	emb := NewEmbedding(vocabSize, dim, true, "Unembedding")
	return UnembeddingLayerFrom(emb, bias)
}

func UnembeddingLayerFrom(emb *Embedding, bias bool) Layer {
	ul := &unembeddingLayer{Embedding: emb}
	if bias {
		ul.bias = make([]*Node, emb.vocabSize)
		for i := range emb.vocabSize {
			ul.bias[i] = NewNode(0, fmt.Sprintf("B_%d", i))
		}
	}
	return ul
}

type unembeddingLayer struct {
	*Embedding
	bias []*Node
}

func (ul *unembeddingLayer) Feed(in []*Node) (out []*Node) {
	if len(in)%ul.dim != 0 {
		panic("input size is not a multiple of unembedding.dim")
	}

	for i := 0; i < len(in); i += ul.dim {
		out = append(out, ul.UnembeddingFeed(in[i:i+ul.dim], ul.bias)...)
	}
	return out
}

func (ul *unembeddingLayer) Parameters() []util.Parameter {
	emb := util.SliceConvert[*Node, util.Parameter](ul.matrix)
	return append(emb, util.SliceConvert[*Node, util.Parameter](ul.bias)...)
}

func (*unembeddingLayer) Name() string { return "unembeddingLayer" }

func PositionalEmbeddingLayer(vocabSize, ctxLen, dim int) Layer {
	pos := NewInputNodeBatch(ctxLen, "P_%d", false)
	for i, p := range pos {
		p.SetV(float64(i))
	}
	return &positionalEmbeddingLayer{
		semantic:   EmbeddingLayer(vocabSize, dim, "TokEmbedding"),
		positional: EmbeddingLayer(ctxLen, dim, "PosEmbedding"),
		pos:        pos,
	}
}

type positionalEmbeddingLayer struct {
	semantic   Layer
	positional Layer
	pos        []*Node
}

func (pel *positionalEmbeddingLayer) Feed(in []*Node) []*Node {
	var (
		semOut = pel.semantic.Feed(in)
		posOut = pel.positional.Feed(pel.pos[:len(in)])
	)
	return VectorAdd(semOut, posOut)
}

func (pel *positionalEmbeddingLayer) Parameters() []util.Parameter {
	return append(pel.semantic.Parameters(), pel.positional.Parameters()...)
}

func (*positionalEmbeddingLayer) Name() string { return "PositionalEmbeddingLayer" }
