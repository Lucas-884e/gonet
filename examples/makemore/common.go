// Package makemore define the common utilities that are shared by all "makemore" examples.
package makemore

import (
	"fmt"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/util"
)

type LazyProbModel interface {
	LazyOutput([]*gonet.Node) func() []float64
}

func LazyProbNext(model LazyProbModel, vocabSize, ctxLen int) func(...int) []float64 {
	var xs []*gonet.Node
	for i := range ctxLen {
		ohe := gonet.NewInputNodeBatch(vocabSize, fmt.Sprintf("X%d_%%d", i))
		xs = append(xs, ohe...)
	}
	lo := model.LazyOutput(xs)

	return func(ctx ...int) []float64 {
		for i, idx := range ctx {
			for j := range vocabSize {
				if x := xs[i*vocabSize+idx]; j == idx {
					x.SetV(1)
				} else {
					x.SetV(0)
				}
			}
		}
		return lo()
	}
}

func GenName(i2c []byte, pnext func(...int) []float64, ctxLen int) string {
	var (
		seq []byte
		ctx = make([]int, ctxLen)
	)
	for {
		idx := util.RandMultinomial(pnext(ctx...))
		if idx == 0 {
			return string(seq)
		}
		seq = append(seq, i2c[idx])
	}
}
