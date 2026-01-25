package graph

import (
	"fmt"
	"testing"

	"github.com/Lucas-884e/gonet/util"
	"github.com/stretchr/testify/assert"
)

func TestBatchLoss(t *testing.T) {
	var (
		a = NewNode(1, "a")
		b = NewNode(2, "b")
		z = BatchLoss(a, b)
	)

	assert.Equal(t, a, BatchLoss(a))

	assert.EqualValues(t, 1.5, z.V())
	assert.Equal(t, "(a+b)×mean", z.Name())

	z.Backward()
	assert.EqualValues(t, 1, z.G())

	sum := z.prev[0]
	assert.EqualValues(t, 0.5, sum.G())
	mean := z.prev[1]
	assert.EqualValues(t, 3, mean.G())

	assert.EqualValues(t, 0.5, a.G())
	assert.EqualValues(t, 0.5, b.G())
}

type mockModel struct {
	w *Node
}

func newMockModel(w float64) *mockModel {
	return &mockModel{w: NewNode(w, "W")}
}

func (m *mockModel) Feed(x []float64) []*Node {
	return []*Node{Multiply(m.w, NewInputNode(x[0], fmt.Sprintf("X(%.3g)", x[0])))}
}

func TestModelLossFunc(t *testing.T) {
	var (
		samples = []util.Sample{
			{X: []float64{1}, Y: []float64{2.1}},
			{X: []float64{0.5}, Y: []float64{0.9}},
			{X: []float64{-0.7}, Y: []float64{-1.5}},
		}
		model  = newMockModel(2)
		lossFn = ModelLossFunc(model, ResidualSumSquaredLoss)
		loss   = lossFn(samples)
		w      = model.w
	)

	loss.Backward()
	fmt.Println(loss)
	g := ((2*1-2.1)*1 + (2*0.5-0.9)*0.5 + (2*(-0.7)-(-1.5))*(-0.7)) / 3
	assert.InDelta(t, g, w.G(), 1e-10)

	w.Learn(5)
	assert.InDelta(t, 2.2, w.V(), 1e-10)
}
