// Package makemore define the common utilities that are shared by all "makemore" examples.
package makemore

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/Lucas-884e/gonet/util"
)

func ReadInCorpus(path string) (corpus [][]byte) {
	scanner := bufio.NewScanner(util.Must1(os.Open(path)))
	for scanner.Scan() {
		if name := strings.TrimSpace(scanner.Text()); name != "" {
			corpus = append(corpus, []byte(name))
		}
	}
	return corpus
}

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

func FormatEmbeddings(embs [][]float64, sep string) string {
	sb := new(strings.Builder)
	sb.WriteString("Embeddings: [\n")
	for _, emb := range embs {
		sb.WriteString("  [")
		for i, x := range emb {
			if i > 0 {
				sb.WriteString(sep)
			}
			fmt.Fprintf(sb, "%.4f", x)
		}
		sb.WriteString("],\n")
	}
	sb.WriteByte(']')
	return sb.String()
}
