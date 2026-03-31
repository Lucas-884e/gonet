package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strings"
	"time"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/util"
)

var data = flag.String("i", "../dataset/names.txt", "Input data file path")

func main() {
	flag.Parse()

	var corpus [][]byte
	scanner := bufio.NewScanner(util.Must1(os.Open(*data)))
	for scanner.Scan() {
		if name := strings.TrimSpace(scanner.Text()); name != "" {
			corpus = append(corpus, []byte(name))
		}
	}
	log.Printf("Corpus size: %d", len(corpus))

	c2i := util.GenVocabFromCorpus(corpus, '.')
	i2c := util.GetIndexToToken(c2i)
	vocabSize := len(c2i)
	log.Printf("Vocabulary size: %d", vocabSize)
	for _, char := range i2c {
		fmt.Printf("%c", char)
	}
	fmt.Println()

	indices := util.CorpusToTokenIndexSequences(corpus, c2i)
	inputs, labels := util.GenInputsAndLabelsFromTokenIndexSequence(indices, c2i['.'])
	samples := util.GenDatasetFromInputsAndLabels(inputs, labels, vocabSize)

	var (
		mlp = constructMLP(vocabSize)
		cfg = util.TrainConfig{
			BatchSize:    500,
			Epochs:       20,
			StopEps:      1e-6,
			LearningRate: 0.3,
		}
		start = time.Now()
	)
	util.ShuffleSamples(samples)
	trainBigramModel(mlp, samples, cfg)
	log.Printf("Training time cost: %s", time.Since(start))
}

func constructMLP(vocabSize int) *gonet.MLP {
	mlp := gonet.NewMLP(vocabSize)
	mlp.AddLayer(vocabSize, gonet.OpSoftmax, false)
	return mlp
}

func trainBigramModel(mlp *gonet.MLP, samples []util.Sample, cfg util.TrainConfig) {
	// Don't do evaluation on the entire dataset, it will eat all your memory.
	const nSamples = 10000
	var (
		delta                 float64
		vocabSize             = len(samples[0].X)
		input, loss           = buildLoss(mlp, vocabSize, nSamples)
		batchInput, batchLoss = buildLoss(mlp, vocabSize, cfg.BatchSize)
		// optimizer             = util.SGDOptimizer(mlp.Parameters(), cfg.LearningRate)
		optimizer = util.DefaultAdamOptimizer(mlp.Parameters(), cfg.LearningRate)
	)

	// Evaluation before training.
	input.Update(samples[:nSamples])
	loss.Forward()
	log.Printf("[After training] total loss: %g", loss.V())

train:
	for ep := 0; ep < cfg.Epochs; ep++ {
		for start := 0; start < len(samples); start += cfg.BatchSize {
			end := min(start+cfg.BatchSize, len(samples))
			batchInput.Update(samples[start:end])
			batchLoss.Backward()

			if delta = optimizer.Learn(); delta < cfg.StopEps && batchLoss.V() < cfg.StopEps {
				log.Printf("* Reached stopping criterion (delta = %g | loss=%g < epsilon=%g).", delta, batchLoss.V(), cfg.StopEps)
				break train
			}
		}

		if ep%5 == 0 || ep+1 == cfg.Epochs {
			log.Printf("[Epoch=%d] Gradient descent delta=%g | loss=%g", ep, delta, batchLoss.V())
		}
	}

	// Evaluation after training.
	loss.Forward()
	log.Printf("[After training] total loss: %g", loss.V())
}

func buildLoss(mlp *gonet.MLP, vocabSize, inputSize int) (input gonet.SampleBatch, loss *gonet.Node) {
	input = gonet.NewSampleBatch(vocabSize, vocabSize, inputSize)
	loss = gonet.ModelLossFunc(mlp, gonet.CrossEntropyLoss)(input)
	return
}
