// Package makemore define the common utilities that are shared by all "makemore" examples.
package makemore

import (
	"github.com/Lucas-884e/gonet/util"
)

func GenName(i2c []byte, pnext func(...int) []float64, ctxLen int) string {
	var (
		seq []byte
		ctx = make([]int, ctxLen)
	)
	for range 100 {
		idx := util.RandMultinomial(pnext(ctx...))
		if idx == 0 {
			break
		}
		ctx = append(ctx[1:], idx)
		seq = append(seq, i2c[idx])
	}
	return string(seq)
}
