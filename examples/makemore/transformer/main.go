package main

import (
	"flag"
	"log"
	"math/rand/v2"
	"os"
	"strconv"
	"time"

	"github.com/LucasInOz/gonet"
	"github.com/LucasInOz/gonet/util"
)

var (
	data        = flag.String("i", "../dataset/shakespeare.txt", "Input data file path")
	interactive = flag.Bool("interactive", false, "Turn on interactive mode")
)

const (
	ctxLen   = 12 // context length
	layerNum = 3  // number of attention layers
	headNum  = 6  // number of attention heads
	embDim   = 96 // embedding space dimension

	batchSize       = 20
	samplesPerEpoch = 500
	learningRate    = 0.0003
)

func main() {
	flag.Parse()

	corpus := util.Must1(os.ReadFile(*data))
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

	var (
		model   = gonet.DecoderOnlyTransformer(vocabSize, ctxLen, layerNum, headNum, embDim)
		predict = func(num int) func(...string) {
			return func(params ...string) {
				if len(params) > 0 {
					num, _ = strconv.Atoi(params[0])
				}
				log.Printf("Generate:\n<|BEGIN|>\n%s\n<|END|>", generate(model, c2i, i2c, num))
			}
		}
	)
	predict(100)()

	cfg := util.TrainConfig{
		BatchSize:        batchSize,
		Epochs:           500,
		LearningRate:     learningRate,
		LogEpochInterval: 500,
		Sampler: func() []util.Sample {
			return randSamples(trainSet, ctxLen, batchSize)
		},
	}
	util.InteractiveTrain(&cfg, *interactive, func() time.Duration {
		var (
			trSamples  = randSamples(trainSet, ctxLen, samplesPerEpoch)
			valSamples = randSamples(valSet, ctxLen, samplesPerEpoch)
		)
		return gonet.Train(model, [][]util.Sample{trSamples, valSamples}, &cfg, gonet.CrossEntropyLoss)
	}, predict(50))

	predict(200)()
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
