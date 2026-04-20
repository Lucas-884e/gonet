package util

import "slices"

type TokenType interface {
	~string | byte
}

func GenVocabFromCorpus[T TokenType](corpus [][]T, eos T) map[T]int {
	tokenSet := make(map[T]struct{})
	for _, seq := range corpus {
		for _, token := range seq {
			if _, ok := tokenSet[token]; !ok {
				tokenSet[token] = struct{}{}
			}
		}
	}

	tokens := make([]T, 0, len(tokenSet))
	for token := range tokenSet {
		tokens = append(tokens, token)
	}
	slices.Sort(tokens)

	vocab := map[T]int{eos: 0}
	for i, token := range tokens {
		vocab[token] = i + 1
	}
	return vocab
}

func GetIndexToToken[T TokenType](vocab map[T]int) []T {
	m := make([]T, len(vocab))
	for token, idx := range vocab {
		m[idx] = token
	}
	return m
}

func TokensToIndexes[T TokenType](tokens []T, vocab map[T]int) (indexes []int) {
	for _, t := range tokens {
		indexes = append(indexes, vocab[t])
	}
	return
}

func GenInputsAndLabelsFromCorpus[T TokenType](corpus [][]T, vocab map[T]int, ctxLen int) ([][]int, []int) {
	var (
		inputs = make([][]int, 0, len(corpus))
		labels = make([]int, 0, len(corpus))
	)
	for _, seq := range corpus {
		ctx := make([]int, ctxLen)
		for _, token := range seq {
			idx, ok := vocab[token]
			if !ok {
				panic("Character '" + string(token) + "' not in alphabet")
			}
			inputs = append(inputs, ctx)
			labels = append(labels, idx)
			ctx = append(ctx[1:], idx)
		}
		inputs = append(inputs, ctx)
		labels = append(labels, 0)
	}
	return inputs, labels
}

func OneHot(index, vocabSize int) []float64 {
	v := make([]float64, vocabSize)
	v[index] = 1.0
	return v
}

func OneHots(indexes []int, vocabSize int) []float64 {
	v := make([]float64, 0, vocabSize*len(indexes))
	for _, idx := range indexes {
		v = append(v, OneHot(idx, vocabSize)...)
	}
	return v
}

func GenDatasetFromInputsAndLabels(inputs [][]int, labels []int) (samples []Sample) {
	for i, input := range inputs {
		x := NumberSliceConvert[int, float64](input)
		y := []float64{float64(labels[i])}
		samples = append(samples, Sample{X: x, Y: y})
	}
	return
}

func GenOneHotDatasetFromInputsAndLabels(inputs [][]int, labels []int, vocabSize int) (samples []Sample) {
	for i, input := range inputs {
		x := OneHots(input, vocabSize)
		y := []float64{float64(labels[i])}
		samples = append(samples, Sample{X: x, Y: y})
	}
	return
}
