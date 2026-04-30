package gonet

import "github.com/Lucas-884e/gonet/util"

func KQVLayer(embDim, headSize int) Layer {
	return &kqvLayer{
		embDim: embDim,
		key:    LinearLayer(embDim, headSize, false),
		query:  LinearLayer(embDim, headSize, false),
		value:  LinearLayer(embDim, headSize, false),
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
	return append(append(ks, qs...), vs...)
}

func (l *kqvLayer) Parameters() []util.Parameter {
	return append(append(l.key.Parameters(), l.query.Parameters()...), l.value.Parameters()...)
}

func (*kqvLayer) Name() string { return "KeyQueryValueLayer" }

func MaskedSelfAttentionLayer(embDim, headSize int) Layer {
	return &maskedSelfAttentionLayer{
		headSize: headSize,
		kqv:      KQVLayer(embDim, headSize),
		linear:   LinearLayer(3*headSize, headSize, false),
	}
}

type maskedSelfAttentionLayer struct {
	headSize int
	kqv      Layer
	linear   Layer // TODO:
}

func (al *maskedSelfAttentionLayer) Feed(in []*Node) (out []*Node) {
	kqv := al.kqv.Feed(in)
	out = al.linear.Feed(kqv) // TODO:
	return out
}

func (al *maskedSelfAttentionLayer) Parameters() []util.Parameter {
	return append(al.kqv.Parameters(), al.linear.Parameters()...)
}

func (*maskedSelfAttentionLayer) Name() string { return "MaskedSelfAttentionLayer" }

func AttentionBlockLayer(embDim, headNum int, buildAttention func(int, int) Layer) Layer {
	if embDim%headNum != 0 {
		panic("embedding dimension must be a multiple of attention head number")
	}

	var (
		headSize = embDim / headNum
		heads    = make([]Layer, headNum)
	)
	for i := range headNum {
		heads[i] = buildAttention(embDim, headSize)
	}

	return &attentionBlockLayer{
		heads: heads,
		ffwd: SequentialModel(
			LinearLayer(embDim, 4*embDim, true),
			ReluLayer(),
			LinearLayer(4*embDim, embDim, true),
		),
	}
}

type attentionBlockLayer struct {
	heads []Layer // Atention heads
	ffwd  Layer   // Feedforward layer
}

func (abl *attentionBlockLayer) Feed(in []*Node) (out []*Node) {
	var values []*Node
	for _, h := range abl.heads {
		values = append(values, h.Feed(in)...)
	}
	return abl.ffwd.Feed(values)
}

func (abl *attentionBlockLayer) Parameters() []util.Parameter {
	var p []util.Parameter
	for _, h := range abl.heads {
		p = append(p, h.Parameters()...)
	}
	return append(p, abl.ffwd.Parameters()...)
}

func (*attentionBlockLayer) Name() string { return "AttentionBlockLayer" }
