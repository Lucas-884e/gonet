package main

import (
	"flag"
	"log"
	"math/rand/v2"
	"os"
	"time"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/util"
)

var (
	data        = flag.String("i", "../dataset/shakespeare.txt", "Input data file path")
	interactive = flag.Bool("interactive", false, "Turn on interactive mode")
)

const (
	ctxLen   = 4  // context length
	layerNum = 1  // number of attention layers
	headNum  = 4  // number of attention heads
	embDim   = 32 // embedding space dimension
)

func main() {
	flag.Parse()

	corpus := util.Must1(os.ReadFile(*data))[:3000]
	log.Printf("First 300 characters from corpus (size=%d): \n<|BEGIN|>\n%s\n<|END|>", len(corpus), corpus[:300])
	c2i := util.GenVocabFromCorpus([][]byte{corpus}, '\n')
	i2c := util.GetIndexToToken(c2i)
	vocabSize := len(c2i)
	log.Printf("Vocabulary (size=%d): %c", vocabSize, i2c)

	inputs := util.TokensToIndexes(corpus, c2i)
	log.Printf("Corpus encoded (first 50 indexes): %v", inputs[:50])

	trainSize := len(inputs) * 9 / 10
	trainSet, valSet := inputs[:trainSize], inputs[trainSize:]
	log.Printf("Training set size=%d, validation set size=%d", trainSize, len(valSet))

	model := gonet.DecoderOnlyTransformer(vocabSize, ctxLen, layerNum, headNum, embDim)
	log.Printf("Generate:\n<|BEGIN|>\n%s\n<|END|>", generate(model, c2i, i2c, 100))

	cfg := util.TrainConfig{
		BatchSize:        20,
		Epochs:           20,
		LearningRate:     0.003,
		LogEpochInterval: 10,
	}
	util.InteractiveTrain(&cfg, *interactive, func() time.Duration {
		samples := randSamples(trainSet, ctxLen, 600)
		return gonet.Train(model, samples, &cfg, gonet.CrossEntropyLoss)
	})

	log.Printf("Generate:\n<|BEGIN|>\n%s\n<|END|>", generate(model, c2i, i2c, 100))
}

func generate(model gonet.Model, c2i map[byte]int, i2c []byte, maxGenTokens int, ctx ...byte) []byte {
	var ctxIdx []float64
	if len(ctx) == 0 {
		ctxIdx = []float64{0}
	} else {
		ctxIdx = util.NumberSliceConvert[int, float64](util.TokensToIndexes(ctx, c2i))
	}

	var genIdx []int
	for range maxGenTokens {
		if len(ctxIdx) > ctxLen {
			ctxIdx = ctxIdx[len(ctxIdx)-ctxLen:]
		}
		var (
			logits = model.Predict(ctxIdx)
			probs  = util.Softmax(1, logits[len(logits)-len(i2c):])
			idx    = util.RandMultinomial(probs)
		)
		genIdx = append(genIdx, idx)
		ctxIdx = append(ctxIdx, float64(idx))
	}

	return util.IndexesToTokens(genIdx, i2c)
}

func randSamples(dataset []int, ctxLen, count int) (samples []util.Sample) {
	for range count {
		r := rand.IntN(len(dataset) - ctxLen)
		samples = append(samples, util.Sample{
			X: util.NumberSliceConvert[int, float64](dataset[r : r+ctxLen]),
			Y: util.NumberSliceConvert[int, float64](dataset[r+1 : r+ctxLen+1]),
		})
	}
	return
}
