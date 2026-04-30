package main

import (
	"flag"
	"fmt"
	"log"
	"time"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/examples/makemore"
	"github.com/Lucas-884e/gonet/util"
)

var (
	data        = flag.String("i", "../dataset/names.txt", "Input data file path")
	interactive = flag.Bool("interactive", false, "Turn on interactive mode")
)

const (
	ctxLen  = 4  // context length
	embDim  = 10 // embedding space dimension
	hidSize = 32 // hidden layer size
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

	inputs, labels := util.GenInputsAndLabelsFromCorpus(corpus, c2i, ctxLen)
	samples := util.GenDatasetFromInputsAndLabels(inputs, labels)
	printSamples(samples[:10], i2c)

	var (
		model = gonet.SequentialModel(
			gonet.EmbeddingLayer(vocabSize, embDim),
			gonet.LinearLayer(embDim*2, hidSize, false),
			gonet.TanhLayer(),
			gonet.LinearLayer(hidSize*2, hidSize, false),
			gonet.TanhLayer(),
			gonet.DisembeddingLayer(vocabSize, hidSize, true),
		)
		pnext = func(in ...int) []float64 {
			xs := util.NumberSliceConvert[int, float64](in)
			return util.Softmax(model.Predict(xs))
		}
	)
	for i := range 10 {
		name := makemore.GenName(i2c, pnext, ctxLen)
		fmt.Printf("[Before training] Generate name (%d | len=%d): %s\n", i+1, len(name), name)
	}

	cfg := util.TrainConfig{
		BatchSize:        32,
		Epochs:           10,
		LearningRate:     0.003,
		LogEpochInterval: 10,
	}
	util.InteractiveTrain(&cfg, *interactive, func() time.Duration {
		return gonet.Train(model, samples, &cfg, gonet.CrossEntropyLoss)
	})

	for i := range 10 {
		name := makemore.GenName(i2c, pnext, ctxLen)
		fmt.Printf("[After training] Generate name (%d | len=%d): %s\n", i+1, len(name), name)
	}
}

func printSamples(samples []util.Sample, i2c []byte) {
	for _, s := range samples {
		for _, idx := range s.X {
			fmt.Printf("%c", i2c[int(idx)])
		}
		fmt.Print(" → ")
		for _, idx := range s.Y {
			fmt.Printf("%c", i2c[int(idx)])
		}
		fmt.Println()
	}
}
