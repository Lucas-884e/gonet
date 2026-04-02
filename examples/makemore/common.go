// Package makemore define the common utilities that are shared by all "makemore" examples.
package makemore

import (
	"fmt"
	"strings"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/util"
)

type ProbModel interface {
	Output(xs []*gonet.Node) []float64
}

func BuildProbMatrix(mlp ProbModel, vocabSize int) (mat [][]float64) {
	for idx := range vocabSize {
		xs := gonet.NewInputNodeBatch(vocabSize, "X_%d")
		xs[idx].SetV(1)
		mat = append(mat, mlp.Output(xs))
	}
	return
}

func FormatProbMatrix(pmat [][]float64) string {
	sb := new(strings.Builder)
	for _, ps := range pmat {
		sb.WriteString("\n  ")
		for _, p := range ps {
			fmt.Fprintf(sb, " %.4f", p)
		}
	}
	return sb.String()
}

func BuildLoss(mlp *gonet.MLP, vocabSize, inputSize int) (input gonet.SampleBatch, loss *gonet.Node) {
	input = gonet.NewSampleBatch(vocabSize, vocabSize, inputSize)
	loss = gonet.ModelLossFunc(mlp, gonet.CrossEntropyLoss)(input)
	return
}

func GenName(i2c []byte, probMat [][]float64) string {
	var (
		idx int
		seq []byte
	)
	for {
		idx = util.RandMultinomial(probMat[idx])
		if idx == 0 {
			return string(seq)
		}
		seq = append(seq, i2c[idx])
	}
}
