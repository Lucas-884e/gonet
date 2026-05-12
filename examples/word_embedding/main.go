package main

import (
	"flag"
	"fmt"
	"log"
	"strings"

	"github.com/LucasInOz/gonet"
	"github.com/LucasInOz/gonet/util"
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
	emb, unemb := trainEmbeddings(samples, vocabSize, 2)

	fmt.Println()
	log.Print("Vocabulary: ", vocab)
	log.Print("Inputs: ", inputs)
	log.Print("Labels: ", labels)
	for _, s := range samples {
		log.Printf("Sample: %+v", s)
	}

	log.Print("Embeddings: -------------------")
	for i := range vocabSize {
		log.Printf("%10s | %s | %s", idxToToken[i], emb.S(i), unemb.S(i))
	}

	godzillaIndex := vocab["Godzilla"]
	ironmanIndex := vocab["Ironman"]
	diff1 := emb.Sub(godzillaIndex, ironmanIndex)
	diff2 := unemb.Sub(godzillaIndex, ironmanIndex)
	log.Printf("(%s - %s) = %.4f | %.4f", idxToToken[godzillaIndex], idxToToken[ironmanIndex], diff1, diff2)
}

func trainEmbeddings(samples []util.Sample, vocabSize, dim int) (emb, unemb *gonet.Embedding) {
	emb = gonet.NewEmbedding(vocabSize, dim, false)
	unemb = gonet.NewEmbedding(vocabSize, dim, true, "Unembedding")

	var (
		model = gonet.SequentialModel(
			gonet.EmbeddingLayerFrom(emb),
			gonet.UnembeddingLayerFrom(unemb, false),
		)
		precision = util.PredictionPrecision(model, samples, isCorrect)
	)
	log.Printf("[Before training] Prediction precision: %g", precision)

	var (
		cfg = util.TrainConfig{
			BatchSize:        1,
			Epochs:           50000,
			LearningRate:     0.01,
			LogEpochInterval: 5000,
		}
		timeCost = gonet.Train(model, samples, &cfg, gonet.CrossEntropyLoss)
	)
	log.Printf("Training time cost: %s", timeCost)

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
