package gonet

import (
	"fmt"
	"math"
	"math/rand/v2"

	"github.com/LucasInOz/gonet/util"
)

type Layer interface {
	FeedForwarder
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
	return []*Node{Linear(in, sl.weights, sl.bias)}
}

func (sl *singleLinear) Parameters() []util.Parameter {
	ps := util.SliceConvert[*Node, util.Parameter](sl.weights)
	if sl.bias != nil {
		ps = append(ps, sl.bias)
	}
	return ps
}

func (sl *singleLinear) Name() string { return "SingleLinear" }

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
