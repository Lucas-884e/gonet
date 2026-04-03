package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/examples/makemore"
	"github.com/Lucas-884e/gonet/util"
)

var data = flag.String("i", "../dataset/names.txt", "Input data file path")

const (
	ctxLen  = 2  // context length
	embDim  = 2  // embedding space dimension
	hidSize = 10 // hidden layer size
)

func main() {
	flag.Parse()

	var corpus [][]byte
	scanner := bufio.NewScanner(util.Must1(os.Open(*data)))
	for scanner.Scan() {
		if name := strings.TrimSpace(scanner.Text()); name != "" {
			corpus = append(corpus, []byte(name))
		}
	}
	corpus = corpus[:5]
	log.Printf("Corpus size: %d", len(corpus))

	c2i := util.GenVocabFromCorpus(corpus, '.')
	i2c := util.GetIndexToToken(c2i)
	vocabSize := len(c2i)
	log.Printf("Vocabulary (size=%d): %c", vocabSize, i2c)

	// indices := util.CorpusToTokenIndexSequences(corpus, c2i, 1)
	inputs, labels := util.GenInputsAndLabelsFromCorpus(corpus, c2i, ctxLen)
	samples := util.GenDatasetFromInputsAndLabels(inputs, labels, ctxLen, vocabSize)
	for _, s := range samples {
		fmt.Println("X:", s.X)
		fmt.Println("Y:", s.Y)
	}

	var (
		model = constructModel(vocabSize)
		pnext = makemore.LazyProbNext(model, vocabSize, ctxLen)
	)
	for i := range 20 {
		fmt.Printf("[Before training] Generate name (%d): %s\n", i+1, makemore.GenName(i2c, pnext, ctxLen))
	}
}

type Model struct {
	embeddingLayer *gonet.MLP
	predictionMLP  *gonet.MLP
}

func (m *Model) LazyOutput(xs []*gonet.Node) func() []float64 {
	var (
		vocabSize    = m.embeddingLayer.InputSize()
		ctxLen       = len(xs) / vocabSize
		ctxEmbedding = make([]*gonet.Node, 0, ctxLen*embDim)
	)
	for i := range ctxLen {
		ohe := xs[i : i+vocabSize]
		embedding := m.embeddingLayer.Feed(ohe)
		ctxEmbedding = append(ctxEmbedding, embedding...)
	}
	return m.predictionMLP.LazyOutput(ctxEmbedding)
}

func constructModel(vocabSize int) *Model {
	embedding := gonet.NewMLP(vocabSize)
	embedding.AddLayer(embDim, gonet.OpNone, false)

	predictor := gonet.NewMLP(embDim * ctxLen)
	predictor.AddLayer(hidSize, gonet.OpTanh, true)
	predictor.AddLayer(vocabSize, gonet.OpSoftmax, true)

	return &Model{
		embeddingLayer: embedding,
		predictionMLP:  predictor,
	}
}
