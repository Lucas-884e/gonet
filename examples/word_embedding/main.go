package main

import (
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/util"
)

var corpus = `Godzilla is great
Ironman is great`

func main() {
	flag.Parse()

	sentences := tokenize(corpus)
	vocab := util.GenVocabFromCorpus(sentences, "<EOS>")
	idxToToken := util.GetIndexToToken(vocab)
	vocabSize := len(vocab)

	inputs, labels := util.GenInputsAndLabelsFromCorpus(sentences, vocab, 1)
	samples := util.GenDatasetFromInputsAndLabels(inputs, labels)
	emb, disemb := trainEmbeddings(samples, vocabSize, 2)

	fmt.Println()
	log.Print("Vocabulary: ", vocab)
	log.Print("Inputs: ", inputs)
	log.Print("Labels: ", labels)
	for _, s := range samples {
		log.Printf("Sample: %+v", s)
	}

	log.Print("Embeddings: -------------------")
	for i := range vocabSize {
		log.Printf("%10s | %s | %s", idxToToken[i], emb.S(i), disemb.S(i))
	}

	godzillaIndex := vocab["Godzilla"]
	ironmanIndex := vocab["Ironman"]
	diff1 := emb.Sub(godzillaIndex, ironmanIndex)
	diff2 := disemb.Sub(godzillaIndex, ironmanIndex)
	log.Printf("(%s - %s) = %.4f | %.4f", idxToToken[godzillaIndex], idxToToken[ironmanIndex], diff1, diff2)
}

func trainEmbeddings(samples []util.Sample, vocabSize, dim int) (emb, disemb *gonet.Embedding) {
	emb = gonet.NewEmbedding(vocabSize, dim)
	disemb = gonet.NewEmbedding(vocabSize, dim)

	model := gonet.SequentialModel(
		gonet.EmbeddingLayer(emb),
		gonet.DisembeddingLayer(disemb, false),
	)
	precision := util.PredictionPrecision(model, samples, isCorrect)
	log.Printf("[Before training] Prediction precision: %g", precision)

	var (
		cfg = util.TrainConfig{
			BatchSize:    1,
			Epochs:       50000,
			StopEps:      1e-12,
			LearningRate: 0.01,
		}
		optimizer = util.DefaultAdamOptimizer(model.Parameters(), cfg.LearningRate)
		lossFn    = gonet.ModelLossFunc(model, gonet.CrossEntropyLoss)
		loss      *gonet.Node
		delta     float64
	)

train:
	for ep := 0; ep < cfg.Epochs; ep++ {
		for start := 0; start < len(samples); start += cfg.BatchSize {
			end := min(start+cfg.BatchSize, len(samples))
			loss = lossFn(samples[start:end])
			loss.Backward()

			if delta = optimizer.Learn(); delta < cfg.StopEps && loss.V() < cfg.StopEps {
				log.Printf("* Reached stopping criterion (delta = %g | loss=%g < epsilon=%g).", delta, loss.V(), cfg.StopEps)
				break train
			}
		}

		if ep%500 == 0 || ep+1 == cfg.Epochs {
			log.Printf("[Epoch=%d] Gradient descent delta=%g | loss=%g", ep, delta, loss.V())
		}
	}

	precision = util.PredictionPrecision(model, samples, isCorrect)
	log.Printf("[After training] Prediction precision: %g", precision)
	return
}

func tokenize(corpus string) (sentences [][]string) {
	for line := range strings.SplitSeq(corpus, "\n") {
		if line = strings.TrimSpace(line); line == "" {
			continue
		}
		sentences = append(sentences, strings.Fields(line))
	}
	return
}

func isCorrect(pred, label []float64) bool {
	var predicted int
	for i, p := range pred {
		if p > pred[predicted] {
			predicted = i
		}
	}
	return predicted == int(label[0])
}
