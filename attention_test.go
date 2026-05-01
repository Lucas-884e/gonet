package gonet

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestSplitKQV(t *testing.T) {
	var (
		kqv = []int{
			0, 1, 2, 3, 4, 5,
			6, 7, 8, 9, 10, 11,
			12, 13, 14, 15, 16, 17,
		}
		ks, qs, vs = splitKQV(kqv, 2)
	)
	assert.Equal(t, [][]int{{0, 1}, {2, 3}, {4, 5}}, ks)
	assert.Equal(t, [][]int{{6, 7}, {8, 9}, {10, 11}}, qs)
	assert.Equal(t, [][]int{{12, 13}, {14, 15}, {16, 17}}, vs)
}

func TestRearrangeMultiHeadOut(t *testing.T) {
	out := []int{
		1, 2, 3, 4, 5, 6, 7, 8,
		9, 10, 11, 12, 13, 14, 15, 16,
		17, 18, 19, 20, 21, 22, 23, 24,
	}
	newOut := rearrangeMultiHeadOut(out, 3, 2)
	assert.Equal(t, []int{
		1, 2, 9, 10, 17, 18,
		3, 4, 11, 12, 19, 20,
		5, 6, 13, 14, 21, 22,
		7, 8, 15, 16, 23, 24,
	}, newOut)
}
