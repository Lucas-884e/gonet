package gonet

import (
	"fmt"
	"math"
	"math/rand/v2"
	"strings"

	"github.com/Lucas-884e/gonet/util"
)

type Layer interface {
	Feed([]*Node) []*Node
	Parameters() []util.Parameter
	Name() string
}

func SingleLinear(inputSize int, bias bool) *singleLinear {
	var (
		sl         = new(singleLinear)
		normFactor = math.Sqrt(float64(inputSize))
	)

	for widx := range inputSize {
		// Initial weights must be normalized, otherwise training won't converge.
		w := rand.NormFloat64() / normFactor
		wn := NewNode(w, fmt.Sprintf("W_%d", widx))
		sl.weights = append(sl.weights, wn)
	}

	if bias {
		sl.bias = NewNode(0, "B")
	}

	return sl
}

type singleLinear struct {
	weights []*Node
	bias    *Node
}

func (sl *singleLinear) Feed(in []*Node) []*Node {
	var wx []*Node
	for i, w := range sl.weights {
		wx = append(wx, Multiply(w, in[i]))
	}
	if sl.bias != nil {
		wx = append(wx, sl.bias)
	}
	sum := Plus(wx...)
	return []*Node{sum}
}

func (sl *singleLinear) Parameters() []util.Parameter {
	ps := util.SliceConvert[*Node, util.Parameter](sl.weights)
	if sl.bias != nil {
		ps = append(ps, sl.bias)
	}
	return ps
}

func (sl *singleLinear) Name() string { return "SingleLinear" }

func (sl *singleLinear) String() string {
	const maxPrint = 10 // Print at most 10 weights
	ws := make([]string, min(len(sl.weights), maxPrint))
	for i, w := range sl.weights {
		ws[i] = fmt.Sprintf("W[%d]=%.6g", i, w.V())
		if i == maxPrint-1 {
			ws = append(ws, "...")
			break
		}
	}
	if sl.bias != nil {
		ws = append(ws, fmt.Sprintf("Bias=%.6g", sl.bias.V()))
	}
	return strings.Join(ws, " | ")
}

func (sl *singleLinear) loadWeights(ws []float64) {
	for i, w := range sl.weights {
		w.v = ws[i]
	}
	if sl.bias != nil {
		sl.bias.v = ws[len(ws)-1]
	}
}

func LinearLayer(fanIn, fanOut int, bias bool) Layer {
	ll := &linearLayer{fanIn: fanIn}
	for range fanOut {
		ll.neurons = append(ll.neurons, SingleLinear(fanIn, bias))
	}
	return ll
}

type linearLayer struct {
	fanIn   int
	neurons []*singleLinear
}

func (ll *linearLayer) Feed(in []*Node) (out []*Node) {
	if len(in)%ll.fanIn != 0 {
		panic("input size is not a multiple of linearLayer.fanIn")
	}

	for i := 0; i < len(in); i += ll.fanIn {
		out = append(out, ll.feed(in[i:i+ll.fanIn])...)
	}
	return out
}

func (ll *linearLayer) feed(in []*Node) (out []*Node) {
	for _, n := range ll.neurons {
		out = append(out, n.Feed(in)...)
	}
	return out
}

func (ll *linearLayer) Parameters() (p []util.Parameter) {
	for _, n := range ll.neurons {
		p = append(p, n.Parameters()...)
	}
	return p
}

func (ll *linearLayer) Name() string { return "LinearLayer" }

func (ll *linearLayer) loadWeights(ws [][]float64) {
	for i, n := range ll.neurons {
		n.loadWeights(ws[i])
	}
}

func ReluLayer() Layer    { return acRelu }
func SigmoidLayer() Layer { return acSigmoid }
func TanhLayer() Layer    { return acTanh }

type activationLayer int32

const (
	acNone activationLayer = iota
	acRelu
	acSigmoid
	acTanh
)

func (al activationLayer) fn() func(*Node) *Node {
	switch al {
	case acNone:
		return Identity
	case acRelu:
		return Relu
	case acSigmoid:
		return Sigmoid
	case acTanh:
		return Tanh
	default:
		panic("unknown activation function")
	}
}

func (al activationLayer) Feed(in []*Node) (out []*Node) {
	f := al.fn()
	for _, n := range in {
		out = append(out, f(n))
	}
	return out
}

func (al activationLayer) Parameters() []util.Parameter { return nil }

func (al activationLayer) Name() string {
	switch al {
	case acNone:
		return "IdentityLayer"
	case acRelu:
		return "ReluLayer"
	case acSigmoid:
		return "SigmoidLayer"
	case acTanh:
		return "TanhLayer"
	default:
		return "Unknown"
	}
}

func SoftmaxLayer(t float64) Layer {
	return &softmaxLayer{temperature: t}
}

type softmaxLayer struct {
	temperature float64
}

func (sl *softmaxLayer) Feed(in []*Node) []*Node {
	return Softmax(sl.temperature, in...)
}

func (sl *softmaxLayer) Parameters() []util.Parameter { return nil }
func (sl *softmaxLayer) Name() string                 { return "SoftmaxLayer" }

func EmbeddingLayer(emb *Embedding) Layer {
	return (*embeddingLayer)(emb)
}

type embeddingLayer Embedding

func (el *embeddingLayer) Feed(in []*Node) (out []*Node) {
	return (*Embedding)(el).EmbeddingFeed(in)
}

func (el *embeddingLayer) Parameters() []util.Parameter {
	return util.SliceConvert[*Node, util.Parameter](el.matrix)
}

func (el *embeddingLayer) Name() string { return "EmbeddingLayer" }

func DisembeddingLayer(emb *Embedding, bias bool) Layer {
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

func (dl *disembeddingLayer) Name() string { return "DisembeddingLayer" }

func LayerNormLayer(dim int) Layer {
	var (
		gamma = make([]*Node, dim)
		beta  = make([]*Node, dim)
	)
	for i := range dim {
		gamma[i] = NewNode(rand.NormFloat64(), fmt.Sprintf("gamma_%d", i))
		beta[i] = NewNode(rand.NormFloat64(), fmt.Sprintf("beta_%d", i))
	}
	return &layerNormLayer{
		gamma: gamma,
		beta:  beta,
		eps:   1e-5,
	}
}

type layerNormLayer struct {
	gamma []*Node
	beta  []*Node
	eps   float64
}

func (lnl *layerNormLayer) Feed(in []*Node) (out []*Node) {
	return LayerNorm(in, lnl.gamma, lnl.beta, lnl.eps)
}

func (lnl *layerNormLayer) Parameters() []util.Parameter {
	return append(util.SliceConvert[*Node, util.Parameter](lnl.gamma),
		util.SliceConvert[*Node, util.Parameter](lnl.beta)...)
}

func (lnl *layerNormLayer) Name() string { return "LayerNormalizationLayer" }
