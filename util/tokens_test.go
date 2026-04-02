package util

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestGenInputsAndLabelsFromCorpus(t *testing.T) {
	var (
		vocab  = map[byte]int{'.': 0, 'a': 1, 'b': 2}
		corpus = [][]byte{{'a', 'b'}, {'b', 'a', 'a', 'b'}}
	)
	{
		inputs, labels := GenInputsAndLabelsFromCorpus(corpus, vocab, 1)
		assert.Equal(t, [][]int{{0}, {1}, {2}, {0}, {2}, {1}, {1}, {2}}, inputs)
		assert.Equal(t, []int{1, 2, 0, 2, 1, 1, 2, 0}, labels)
	}
	{
		inputs, labels := GenInputsAndLabelsFromCorpus(corpus, vocab, 3)
		assert.Equal(t, [][]int{
			{0, 0, 0},
			{0, 0, 1},
			{0, 1, 2},
			{0, 0, 0},
			{0, 0, 2},
			{0, 2, 1},
			{2, 1, 1},
			{1, 1, 2},
		}, inputs)
		assert.Equal(t, []int{1, 2, 0, 2, 1, 1, 2, 0}, labels)
	}
}
