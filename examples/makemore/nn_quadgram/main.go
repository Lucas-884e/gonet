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
	interactive = flag.Bool("interactive", false, "Turn on interactive mode")
)

const (
	ctxLen  = 3  // context length
	embDim  = 10 // embedding space dimension
	hidSize = 36 // hidden layer size
)

type Model struct {
	vocabSize int

	emb *gonet.Embedding
	m   gonet.Model
}

func newModel(vocabSize int) *Model {
	emb := gonet.NewEmbedding(vocabSize, embDim)
	m := gonet.SequentialModel(
		gonet.EmbeddingLayerFrom(emb),
		gonet.LinearLayer(embDim*ctxLen, hidSize, true),
		gonet.TanhLayer(),
		gonet.UnembeddingLayer(vocabSize, hidSize, true),
	)

	return &Model{
		vocabSize: vocabSize,
		emb:       emb,
		m:         m,
	}
}

func (m *Model) PredictNextProbs(in ...int) []float64 {
	xs := util.NumberSliceConvert[int, float64](in)
	return util.Softmax(1, m.m.Predict(xs))
}

func (m *Model) Embeddings() (embs [][]float64) {
	for idx := range m.vocabSize {
		v := gonet.NodeValues(m.emb.E(idx))
		embs = append(embs, []float64(v))
	}
	return embs
}

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

	model := newModel(vocabSize)
	for i := range 20 {
		name := makemore.GenName(i2c, model.PredictNextProbs, ctxLen)
		fmt.Printf("[Before training] Generate name (%d | len=%d): %s\n", i+1, len(name), name)
	}

	cfg := util.TrainConfig{
		BatchSize:        32,
		Epochs:           10,
		LearningRate:     0.001,
		LogEpochInterval: 10,
	}
	util.InteractiveTrain(&cfg, *interactive, func() time.Duration {
		return gonet.Train(model.m, samples, &cfg, gonet.CrossEntropyLoss)
	})

	fmt.Println(makemore.FormatEmbeddings(model.Embeddings(), ", "))

	for i := range 20 {
		name := makemore.GenName(i2c, model.PredictNextProbs, ctxLen)
		fmt.Printf("[After training] Generate name (%d | len=%d): %s\n", i+1, len(name), name)
	}
}
