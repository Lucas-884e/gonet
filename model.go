package gonet

import (
	"fmt"

	"github.com/Lucas-884e/gonet/util"
)

type Model interface {
	FeedForwarder
	util.Predictor
	Parameters() []util.Parameter
}

func LinearModel(fanIn, fanOut int, bias bool) Model {
	return SequentialModel(LinearLayer(fanIn, fanOut, bias))
}

func EmbeddingModel(vocabSize, dim int) Model {
	emb := NewEmbedding(vocabSize, dim)
	return SequentialModel(EmbeddingLayer(emb))
}

func SequentialModel(layers ...Layer) Model {
	return &sequentialModel{
		layers: layers,
	}
}

type sequentialModel struct {
	layers []Layer
	input  []*Node
	output []*Node
}

func (sm *sequentialModel) Predict(xs []float64) []float64 {
	if len(sm.input) >= len(xs) {
		for i, x := range xs {
			sm.input[i].SetV(x)
		}
	} else {
		sm.input = make([]*Node, len(xs))
		for i, x := range xs {
			sm.input[i] = NewInputNodeNoGrad(x, fmt.Sprintf("X_%d", i))
		}
		sm.output = sm.Feed(sm.input)
	}

	end := len(xs) * len(sm.output) / len(sm.input)
	for _, out := range sm.output[:end] {
		out.Forward()
	}
	return NodeValues(sm.output[:end])
}

func (sm *sequentialModel) Feed(input []*Node) []*Node {
	out := input
	for _, l := range sm.layers {
		out = l.Feed(out)
	}
	return out
}

func (sm *sequentialModel) Parameters() (p []util.Parameter) {
	for _, l := range sm.layers {
		p = append(p, l.Parameters()...)
	}
	return p
}
