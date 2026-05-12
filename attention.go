package gonet

import (
	"fmt"
	"math"

	"github.com/LucasInOz/gonet/util"
)

func KQVLayer(embDim, headSize, headIdx int) Layer {
	return &kqvLayer{
		embDim: embDim,
		key:    LinearLayer(embDim, headSize, false, fmt.Sprintf("K%d", headIdx)),
		query:  LinearLayer(embDim, headSize, false, fmt.Sprintf("Q%d", headIdx)),
		value:  LinearLayer(embDim, headSize, false, fmt.Sprintf("V%d", headIdx)),
	}
}

type kqvLayer struct {
	embDim int

	key   Layer
	query Layer
	value Layer
}

func (l *kqvLayer) Feed(in []*Node) (out []*Node) {
	var (
		ks = l.key.Feed(in)
		qs = l.query.Feed(in)
		vs = l.value.Feed(in)
	)
	return append(append(append(out, ks...), qs...), vs...)
}

func (l *kqvLayer) Parameters() []util.Parameter {
	return append(append(l.key.Parameters(), l.query.Parameters()...), l.value.Parameters()...)
}

func (*kqvLayer) Name() string { return "KeyQueryValueLayer" }

func splitKQV[T any](kqv []T, hs int) (ks, qs, vs [][]T) {
	size := len(kqv) / 3
	for i := 0; i < size; i += hs {
		var (
			k = make([]T, hs)
			q = make([]T, hs)
			v = make([]T, hs)
		)
		for j := range hs {
			k[j] = kqv[i+j]
			q[j] = kqv[size+i+j]
			v[j] = kqv[2*size+i+j]
		}
		ks = append(ks, k)
		qs = append(qs, q)
		vs = append(vs, v)
	}
	return
}

func MaskedAttention(ks, qs, vs [][]*Node) []*Node {
	var (
		multiply = func(a, b []*Node) *Node {
			return DotProduct(a, b)
		}

		temperature = math.Sqrt(float64(len(ks[0])))
		softmax     = func(x []*Node) []*Node {
			return Softmax(temperature, x...)
		}
	)
	return util.MaskedAttention(ks, qs, vs, multiply, softmax)
}

func MaskedSelfAttentionLayer(embDim, headSize, headIdx int) Layer {
	return &maskedSelfAttentionLayer{
		headSize: headSize,
		kqv:      KQVLayer(embDim, headSize, headIdx),
	}
}

type maskedSelfAttentionLayer struct {
	headSize int
	kqv      Layer
}

func (al *maskedSelfAttentionLayer) Feed(in []*Node) []*Node {
	ks, qs, vs := splitKQV(al.kqv.Feed(in), al.headSize)
	return MaskedAttention(ks, qs, vs)
}

func (al *maskedSelfAttentionLayer) Parameters() []util.Parameter {
	return al.kqv.Parameters()
}

func (*maskedSelfAttentionLayer) Name() string { return "MaskedSelfAttentionLayer" }

func rearrangeMultiHeadOut[T any](out []T, headNum, headSize int) []T {
	var (
		newOut = make([]T, len(out))
		step   = len(out) / headNum
		offset int
	)
	for i := 0; i < step; i += headSize {
		for j := i; j < len(out); j += step {
			copy(newOut[offset:], out[j:j+headSize])
			offset += headSize
		}
	}
	return newOut
}

func MultiHeadAttentionLayer(embDim, headNum int, buildAttention func(int, int, int) Layer) Layer {
	if embDim%headNum != 0 {
		panic("embedding dimension must be a multiple of attention head number")
	}

	var (
		headSize = embDim / headNum
		heads    = make([]Layer, headNum)
	)
	for i := range headNum {
		heads[i] = buildAttention(embDim, headSize, i)
	}

	return &multiHeadAttentionLayer{
		headSize:   headSize,
		heads:      heads,
		projection: LinearLayer(embDim, embDim, true),
	}
}

type multiHeadAttentionLayer struct {
	headSize   int
	heads      []Layer
	projection Layer
}

func (mhal *multiHeadAttentionLayer) Feed(in []*Node) []*Node {
	var headOut []*Node
	for _, h := range mhal.heads {
		headOut = append(headOut, h.Feed(in)...)
	}

	projIn := rearrangeMultiHeadOut(headOut, len(mhal.heads), mhal.headSize)
	return mhal.projection.Feed(projIn)
}

func (mhal *multiHeadAttentionLayer) Parameters() (p []util.Parameter) {
	for _, h := range mhal.heads {
		p = append(p, h.Parameters()...)
	}
	return append(p, mhal.projection.Parameters()...)
}

func (*multiHeadAttentionLayer) Name() string { return "MultiHeadAttentionLayer" }

func AttentionBlockLayer(embDim, headNum int, buildAttention func(int, int, int) Layer) Layer {
	return &attentionBlockLayer{
		attention: MultiHeadAttentionLayer(embDim, headNum, buildAttention),
		ffwd: SequentialModel(
			LinearLayer(embDim, 4*embDim, true),
			ReluLayer(),
			LinearLayer(4*embDim, embDim, true),
		),
		norm1: LayerNormLayer(embDim),
		norm2: LayerNormLayer(embDim),
	}
}

type attentionBlockLayer struct {
	attention Layer // Multi-Head Atention
	ffwd      Layer // Feedforward layer
	norm1     Layer // Layer Normalization layer before attention
	norm2     Layer // Layer Normalization layer before feedforward
}

func (abl *attentionBlockLayer) Feed(in []*Node) (out []*Node) {
	var (
		values = abl.attention.Feed(abl.norm1.Feed(in))
		ffwdIn = VectorAdd(in, values)
	)

	ffwdIn = abl.norm1.Feed(ffwdIn)
	return VectorAdd(ffwdIn, abl.ffwd.Feed(ffwdIn))
}

func (abl *attentionBlockLayer) Parameters() []util.Parameter {
	p := abl.attention.Parameters()
	p = append(p, abl.ffwd.Parameters()...)
	p = append(p, abl.norm1.Parameters()...)
	p = append(p, abl.norm2.Parameters()...)
	return p
}

func (*attentionBlockLayer) Name() string { return "AttentionBlockLayer" }
