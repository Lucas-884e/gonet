package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/LucasInOz/gonet"
	"github.com/LucasInOz/gonet/examples/makemore"
	"github.com/LucasInOz/gonet/util"
)

var (
	data        = flag.String("i", "../dataset/names.txt", "Input data file path")
	useLinear   = flag.Bool("linear", false, "Use linear model (otherwise use embedding model)")
	interactive = flag.Bool("interactive", false, "Turn on interactive mode")
)

func main() {
	flag.Parse()

	corpus := makemore.ReadInCorpus(*data)
	// For quick sanity check.
	// corpus = corpus[:5]
	log.Printf("Corpus size: %d", len(corpus))

	c2i := util.GenVocabFromCorpus(corpus, '.')
	i2c := util.GetIndexToToken(c2i)
	vocabSize := len(c2i)
	log.Printf("Vocabulary (size=%d): %c", vocabSize, i2c)

	inputs, labels := util.GenInputsAndLabelsFromCorpus(corpus, c2i, 1)
	var (
		samples []util.Sample
		model   gonet.Model
	)
	if *useLinear {
		samples = util.GenOneHotDatasetFromInputsAndLabels(inputs, labels, vocabSize)
		model = gonet.LinearModel(vocabSize, vocabSize, false)
	} else {
		samples = util.GenDatasetFromInputsAndLabels(inputs, labels)
		model = gonet.EmbeddingModel(vocabSize, vocabSize)
	}

	var (
		pmat  = buildProbMatrix(model, vocabSize)
		pnext = func(ctx ...int) []float64 { return pmat[ctx[0]] }
	)
	for i := range 20 {
		name := makemore.GenName(i2c, pnext, 1)
		fmt.Printf("[Before training] Generate name (%d): %s\n", i+1, name)
	}

	cfg := util.TrainConfig{
		BatchSize:        min(100, len(samples)),
		Epochs:           10,
		StopEps:          1e-8,
		LearningRate:     0.001,
		LogEpochInterval: 10,
	}
	util.InteractiveTrain(&cfg, *interactive, func() time.Duration {
		return gonet.Train(model, samples, &cfg, gonet.CrossEntropyLoss)
	}, nil)

	pmat = buildProbMatrix(model, vocabSize)
	fmt.Printf("[After training] Probability matrix: %s\n", makemore.FormatEmbeddings(pmat, " "))
	for i := range 20 {
		name := makemore.GenName(i2c, pnext, 1)
		fmt.Printf("[After training] Generate name (%d): %s\n", i+1, name)
	}
}

func buildProbMatrix(m gonet.Model, vocabSize int) (mat [][]float64) {
	for idx := range vocabSize {
		var xs []float64
		if *useLinear {
			xs = util.OneHot(idx, vocabSize)
		} else {
			xs = []float64{float64(idx)}
		}
		mat = append(mat, util.Softmax(1, m.Predict(xs)))
	}
	return mat
}
