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
	samples := util.GenDatasetFromInputsAndLabels(inputs, labels, 1, vocabSize)
	embeddings := trainWordEmbedding(samples, 2)

	fmt.Println()
	log.Print("Vocabulary: ", vocab)
	log.Print("Inputs: ", inputs)
	log.Print("Labels: ", labels)
	for _, s := range samples {
		log.Printf("Sample: %+v", s)
	}

	log.Print("Embeddings: -------------------")
	for i, embedding := range embeddings {
		log.Printf("%10s | %s | %s", idxToToken[i], embedding[0], embedding[1])
	}

	godzillaIndex := vocab["Godzilla"]
	ironmanIndex := vocab["Ironman"]
	diff1 := embeddings[godzillaIndex][0].Sub(embeddings[ironmanIndex][0])
	diff2 := embeddings[godzillaIndex][1].Sub(embeddings[ironmanIndex][1])
	log.Printf("(%s - %s) = %s | %s", idxToToken[1], idxToToken[4], diff1, diff2)
}

type Embedding []float64

func (emb Embedding) Sub(other Embedding) Embedding {
	res := make(Embedding, len(emb))
	for i, v := range emb {
		res[i] = v - other[i]
	}
	return res
}

func (emb Embedding) String() string {
	sb := new(strings.Builder)
	sb.WriteByte('[')
	for i, v := range emb {
		if i > 0 {
			sb.WriteString(", ")
		}
		fmt.Fprintf(sb, "%.6g", v)
	}
	sb.WriteByte(']')
	return sb.String()
}

func trainWordEmbedding(samples []util.Sample, dim int) [][2]Embedding {
	vocabSize := len(samples[0].X)
	mlp := gonet.NewMLP(vocabSize)
	mlp.AddLayer(dim, gonet.OpNone, false)
	mlp.AddLayer(vocabSize, gonet.OpSoftmax, false)
	// fmt.Println(mlp)
	log.Printf("Prediction precision before training: %g", PredictionPrecision(mlp, samples))

	var (
		cfg = util.TrainConfig{
			BatchSize:    1,
			Epochs:       50000,
			StopEps:      1e-12,
			LearningRate: 0.01,
		}
		batchInput = gonet.NewSampleBatch(vocabSize, vocabSize, cfg.BatchSize)
		lossFn     = gonet.ModelLossFunc(mlp, gonet.CrossEntropyLoss)
		loss       = lossFn(batchInput)
		optimizer  = util.DefaultAdamOptimizer(mlp.Parameters(), cfg.LearningRate)
		delta      float64
	)

train:
	for ep := 0; ep < cfg.Epochs; ep++ {
		for start := 0; start < len(samples); start += cfg.BatchSize {
			end := min(start+cfg.BatchSize, len(samples))
			batchInput.Update(samples[start:end])
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

	// fmt.Println(mlp)
	log.Printf("Prediction precision after training: %g", PredictionPrecision(mlp, samples))
	return wordEmbeddings(mlp, vocabSize, dim)
}

func wordEmbeddings(model *gonet.MLP, vocabSize, dimension int) [][2]Embedding {
	layers := model.L()
	l1, l2 := layers[0], layers[1]

	res := make([][2]Embedding, vocabSize)
	// Get input-layer-to-hidden-layer embedding.
	for i := range vocabSize {
		res[i][0] = make([]float64, dimension)
		for j, n := range l1.N() {
			res[i][0][j] = n.W()[i].V()
		}
	}
	// Get hidden-layer-to-output-layer embedding.
	for i, n := range l2.N() {
		res[i][1] = make([]float64, dimension)
		for j, w := range n.W() {
			res[i][1][j] = w.V()
		}
	}
	return res
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

func PredictionPrecision(model *gonet.MLP, testSet []util.Sample) float32 {
	var (
		correctCount int
		input        = gonet.NewInputNodeBatch(len(testSet[0].X), "X_%d")
		predicted    = model.Feed(input)
	)
	for _, sample := range testSet {
		for i, x := range sample.X {
			input[i].SetV(x)
		}
		for _, pred := range predicted {
			pred.Forward()
		}
		if isCorrect(gonet.NodeValues(predicted), sample.Y) {
			correctCount++
		}
	}
	return float32(correctCount) / float32(len(testSet))
}

func isCorrect(pred, label []float64) bool {
	var (
		predicted int
		actual    int
	)
	for i, p := range pred {
		if p > pred[predicted] {
			predicted = i
		}
	}
	for i, a := range label {
		if a > label[actual] {
			actual = i
		}
	}
	return actual == predicted
}
