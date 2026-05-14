package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"
	"time"

	"github.com/LucasInOz/gonet"
	"github.com/LucasInOz/gonet/examples/makemore"
	"github.com/LucasInOz/gonet/util"
)

var (
	data        = flag.String("i", "../dataset/names.txt", "Input data file path")
	interactive = flag.Bool("interactive", false, "Turn on interactive mode")
)

const (
	ctxLen  = 4  // context length
	embDim  = 10 // embedding space dimension
	hidSize = 24 // hidden layer size
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
	log.Printf("Training sample count=%d with a few examples:", len(samples))
	printSamples(samples[:10], i2c)

	var (
		model = gonet.SequentialModel(
			gonet.EmbeddingLayer(vocabSize, embDim),
			gonet.LinearLayer(embDim*2, hidSize, false),
			gonet.TanhLayer(),
			gonet.LinearLayer(hidSize*2, hidSize, false),
			gonet.TanhLayer(),
			gonet.UnembeddingLayer(hidSize, vocabSize, true),
		)
		pnext = func(in ...int) []float64 {
			xs := util.NumberSliceConvert[int, float64](in)
			return util.Softmax(1, model.Predict(xs))
		}
		predict = func(num int, stage string) func(...string) {
			return func(params ...string) {
				if len(params) > 0 {
					num, _ = strconv.Atoi(params[0])
				}
				for i := range num {
					name := makemore.GenName(i2c, pnext, ctxLen)
					fmt.Printf("[%s] Generate name (%d | len=%d): %s\n", stage, i+1, len(name), name)
				}
			}
		}
	)
	predict(10, "Before training")()

	cfg := util.TrainConfig{
		BatchSize:        32,
		Epochs:           10,
		LearningRate:     0.001,
		LogEpochInterval: 10,
	}
	util.InteractiveTrain(&cfg, *interactive, func() time.Duration {
		return gonet.Train(model, [][]util.Sample{samples}, &cfg, gonet.CrossEntropyLoss)
	}, predict(5, "During training"))

	predict(20, "After training")()
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
