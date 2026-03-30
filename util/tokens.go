package util

import "slices"

func GenVocabFromCorpus[T ~string | byte](corpus [][]T, eos T) map[T]int {
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

func GetIndexToToken[T ~string | byte](vocab map[T]int) []T {
	m := make([]T, len(vocab))
	for token, idx := range vocab {
		m[idx] = token
	}
	return m
}

func CorpusToTokenIndexSequences[T ~string | byte](corpus [][]T, vocab map[T]int) (indexes []int) {
	for _, seq := range corpus {
		for _, token := range seq {
			idx, ok := vocab[token]
			if !ok {
				panic("Character '" + string(token) + "' not in alphabet")
			}
			indexes = append(indexes, idx)
		}
		indexes = append(indexes, 0)
	}
	return
}

func GenInputsAndLabelsFromTokenIndexSequence(indexes []int, eosIndex int) ([]int, []int) {
	var (
		inputs = make([]int, 0, len(indexes))
		labels = make([]int, 0, len(indexes))
	)
	for i, idx := range indexes {
		if idx == eosIndex {
			continue
		}
		inputs = append(inputs, idx)
		labels = append(labels, indexes[i+1])
	}
	return inputs, labels
}

func OneHot(index, vocabSize int) []float64 {
	v := make([]float64, vocabSize)
	v[index] = 1.0
	return v
}

func GenDatasetFromInputsAndLabels(inputs, labels []int, vocabSize int) (samples []Sample) {
	for i, input := range inputs {
		x := OneHot(input, vocabSize)
		y := OneHot(labels[i], vocabSize)
		samples = append(samples, Sample{X: x, Y: y})
	}
	return
}
