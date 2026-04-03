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
	// For quick validation.
	// corpus = corpus[:5]
	log.Printf("Corpus size: %d", len(corpus))

	c2i := util.GenVocabFromCorpus(corpus, '.')
	i2c := util.GetIndexToToken(c2i)
	vocabSize := len(c2i)
	log.Printf("Vocabulary (size=%d): %c", vocabSize, i2c)

	// indices := util.CorpusToTokenIndexSequences(corpus, c2i, 1)
	inputs, labels := util.GenInputsAndLabelsFromCorpus(corpus, c2i, 1)
	samples := util.GenDatasetFromInputsAndLabels(inputs, labels, 1, vocabSize)

	var (
		mlp  = constructMLP(vocabSize)
		pmat = buildProbMatrix(mlp, vocabSize)
	)
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
	for start := time.Now(); ; start = time.Now() {
		// util.ShuffleSamples(samples)
		trainBigramModel(mlp, samples, cfg)
		log.Printf("Training time cost: %s", time.Since(start))

		pmat = buildProbMatrix(mlp, vocabSize)
		log.Printf("[During training] Probability matrix: %s", formatProbMatrix(pmat))

		fmt.Println("Continue training?\n  (q, quit, exit) exit;\n  (float integer) learning_rate -> float, training epochs -> integer;\n  (otherwise) continue.")
		input, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalf("Read input error: %v", err)
		}
		switch input = strings.TrimSpace(input); input {
		case "q", "quit", "exit":
			break training
		default:
			// Input format:
			// (Case 1) 1.5
			// - Change learning rate to 1.5
			// (Case 2) 1.5 500
			// - Change learning rate to 1.5 and epochs to 500
			fields := strings.Fields(input)
			if len(fields) > 0 {
				if lr, err := strconv.ParseFloat(fields[0], 64); err == nil {
					cfg.LearningRate = lr
					fmt.Printf("Learning rate changed to: %g\n", lr)
				}
			}
			if len(fields) > 1 {
				if ep, err := strconv.Atoi(fields[1]); err == nil {
					cfg.Epochs = ep
					fmt.Printf("Training epochs changed to: %d\n", ep)
				}
			}
		}
	}

	pmat = buildProbMatrix(mlp, vocabSize)
	fmt.Printf("[After training] Probability matrix: %s\n", formatProbMatrix(pmat))
	for i := range 20 {
		fmt.Printf("[After training] Generate name (%d): %s\n", i+1, genName(i2c, pmat))
	}
}

func constructMLP(vocabSize int) *gonet.MLP {
	mlp := gonet.NewMLP(vocabSize)
	mlp.AddLayer(vocabSize, gonet.OpSoftmax, false)
	return mlp
}

func trainBigramModel(mlp *gonet.MLP, samples []util.Sample, cfg util.TrainConfig) {
	// Don't do evaluation on the entire dataset, it will eat all your memory.
	nSamples := min(10000, len(samples))

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
	log.Printf("[Before training] total loss: %g", loss.V())

	for ep := 0; ep < cfg.Epochs; ep++ {
		for start := 0; start < len(samples); start += cfg.BatchSize {
			end := min(start+cfg.BatchSize, len(samples))
			batchInput.Update(samples[start:end])
			batchLoss.Backward()
			delta = optimizer.Learn()
		}

		if ep%10 == 0 || ep+1 == cfg.Epochs {
			log.Printf("[Epoch=%d] Gradient descent delta=%g | loss=%g", ep, delta, batchLoss.V())
		}
	}

	// Evaluation after training.
	loss.Forward()
	log.Printf("[After training] total loss: %g", loss.V())
}

func genName(i2c []byte, probMat [][]float64) string {
	var (
		idx int
		seq []byte
	)
	for {
		idx = util.RandMultinomial(probMat[idx])
		if idx == 0 {
			return string(seq)
		}
		seq = append(seq, i2c[idx])
	}
}

func buildLoss(mlp *gonet.MLP, vocabSize, inputSize int) (input gonet.SampleBatch, loss *gonet.Node) {
	input = gonet.NewSampleBatch(vocabSize, vocabSize, inputSize)
	loss = gonet.ModelLossFunc(mlp, gonet.CrossEntropyLoss)(input)
	return
}

func buildProbMatrix(mlp *gonet.MLP, vocabSize int) (mat [][]float64) {
	for idx := range vocabSize {
		xs := gonet.NewInputNodeBatch(vocabSize, "X_%d")
		xs[idx].SetV(1)
		mat = append(mat, mlp.Output(xs))
	}
	return
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
