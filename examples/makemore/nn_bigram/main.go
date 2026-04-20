package main

import (
	"bufio"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/util"
)

var (
	data        = flag.String("i", "../dataset/names.txt", "Input data file path")
	useLinear   = flag.Bool("linear", false, "Use linear model (otherwise use embedding model)")
	interactive = flag.Bool("interactive", false, "Turn on interactive mode")
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
	// For quick sanity check.
	// corpus = corpus[:5]
	log.Printf("Corpus size: %d", len(corpus))

	c2i := util.GenVocabFromCorpus(corpus, '.')
	i2c := util.GetIndexToToken(c2i)
	vocabSize := len(c2i)
	log.Printf("Vocabulary (size=%d): %c", vocabSize, i2c)

	inputs, labels := util.GenInputsAndLabelsFromCorpus(corpus, c2i, 1)
	var (
		samples []util.Sample
		model   gonet.Model
	)
	if *useLinear {
		samples = util.GenOneHotDatasetFromInputsAndLabels(inputs, labels, vocabSize)
		model = gonet.LinearModel(vocabSize, vocabSize, false)
	} else {
		samples = util.GenDatasetFromInputsAndLabels(inputs, labels)
		model = gonet.EmbeddingModel(vocabSize, vocabSize)
	}

	pmat := buildProbMatrix(model, vocabSize)
	for i := range 20 {
		fmt.Printf("[Before training] Generate name (%d): %s\n", i+1, genName(i2c, pmat))
	}

	var (
		reader = bufio.NewReader(os.Stdin)
		cfg    = util.TrainConfig{
			BatchSize:    min(1000, len(samples)),
			Epochs:       10,
			StopEps:      1e-8,
			LearningRate: 0.001,
		}
	)
training:
	for {
		fmt.Printf("Mini-batch size: %d\nTraining epochs: %d\nLearning rate: %g\n", cfg.BatchSize, cfg.Epochs, cfg.LearningRate)
		timeCost := trainModel(model, samples, cfg)
		log.Printf("Training time cost: %s", timeCost)

		pmat = buildProbMatrix(model, vocabSize)
		log.Printf("[During training] Probability matrix: %s", formatProbMatrix(pmat))

		if !*interactive {
			break
		}

		fmt.Println("Continue training?\n  (q, quit, exit) exit;\n  (integer float) training epochs -> integer, learning_rate -> float;\n  (otherwise) continue.")
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalf("Read input error: %v", err)
		}
		switch input = strings.TrimSpace(input); input {
		case "q", "quit", "exit":
			break training
		default:
			// Input format:
			// (Case 1) 200
			// - Change training epochs to 200
			// (Case 2) 500 0.05
			// - Change training epochs to 500 and learning rate to 0.05
			fields := strings.Fields(input)
			if len(fields) > 0 {
				if ep, err := strconv.Atoi(fields[0]); err == nil {
					cfg.Epochs = ep
				}
			}
			if len(fields) > 1 {
				if lr, err := strconv.ParseFloat(fields[1], 64); err == nil {
					cfg.LearningRate = lr
				}
			}
		}
	}

	pmat = buildProbMatrix(model, vocabSize)
	fmt.Printf("[After training] Probability matrix: %s\n", formatProbMatrix(pmat))
	for i := range 20 {
		fmt.Printf("[After training] Generate name (%d): %s\n", i+1, genName(i2c, pmat))
	}
}

func trainModel(model gonet.Model, samples []util.Sample, cfg util.TrainConfig) time.Duration {
	nSamples := len(samples)
	if *useLinear {
		// Don't do evaluation on the entire dataset when using LinearModel, it
		// will eat all your memory and result in nothing.
		nSamples = 10000
	}

	var (
		optimizer   = util.DefaultAdamOptimizer(model.Parameters(), cfg.LearningRate)
		totalLossFn = gonet.ModelLossFunc(model, gonet.CrossEntropyLoss)
		lossFn      = gonet.ModelLossFunc(model, gonet.CrossEntropyLoss)
		loss        *gonet.Node
		delta       float64
	)

	// Evaluation before training.
	totalLoss := totalLossFn(samples[:nSamples])
	log.Printf("[Before training] total loss: %g", totalLoss.V())

	start := time.Now()
	for ep := 0; ep < cfg.Epochs; ep++ {
		util.ShuffleSamples(samples)
		for start := 0; start < len(samples); start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > len(samples) {
				break // Ignore samples left that cannot form a mini-batch.
			}
			loss = lossFn(samples[start:end])
			loss.Backward()
			delta = optimizer.Learn()
		}

		if ep%10 == 0 || ep+1 == cfg.Epochs {
			log.Printf("[Epoch=%d] Gradient descent delta=%g | loss=%g", ep, delta, loss.V())
		}
	}
	timeCost := time.Since(start)

	// Evaluation after training.
	totalLoss.Forward()
	log.Printf("[After training] total loss: %g", totalLoss.V())
	return timeCost
}

func genName(i2c []byte, probMat [][]float64) string {
	var (
		idx int
		seq []byte
	)
	for range 100 { // Cutoff at 100 to avoid infinite cycle.
		idx = util.RandMultinomial(probMat[idx])
		if idx == 0 {
			break
		}
		seq = append(seq, i2c[idx])
	}
	return string(seq)
}

func buildProbMatrix(m gonet.Model, vocabSize int) (mat [][]float64) {
	for idx := range vocabSize {
		var xs []float64
		if *useLinear {
			xs = util.OneHot(idx, vocabSize)
		} else {
			xs = []float64{float64(idx)}
		}
		mat = append(mat, util.Softmax(m.Predict(xs)))
	}
	return mat
}

func formatProbMatrix(pmat [][]float64) string {
	sb := new(strings.Builder)
	for _, ps := range pmat {
		sb.WriteString("\n  ")
		for _, p := range ps {
			fmt.Fprintf(sb, " %.4f", p)
		}
	}
	return sb.String()
}
