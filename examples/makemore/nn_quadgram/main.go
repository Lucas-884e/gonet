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
	"github.com/Lucas-884e/gonet/examples/makemore"
	"github.com/Lucas-884e/gonet/util"
)

var (
	data        = flag.String("i", "../dataset/names.txt", "Input data file path")
	interactive = flag.Bool("interactive", false, "Turn on interactive mode")
)

const (
	ctxLen  = 3  // context length
	embDim  = 2  // embedding space dimension
	hidSize = 10 // hidden layer size
)

type Model struct {
	vocabSize int

	emb *gonet.Embedding
	m   gonet.Model
}

func newModel(vocabSize int) *Model {
	emb := gonet.NewEmbedding(vocabSize, embDim)
	disemb := gonet.NewEmbedding(vocabSize, hidSize)
	m := gonet.SequentialModel(
		gonet.EmbeddingLayer(emb),
		gonet.LinearLayer(embDim*ctxLen, hidSize, true),
		gonet.TanhLayer(),
		gonet.DisembeddingLayer(disemb, true),
	)

	return &Model{
		vocabSize: vocabSize,
		emb:       emb,
		m:         m,
	}
}

func (m *Model) PredictNextProbs(in ...int) []float64 {
	xs := util.NumberSliceConvert[int, float64](in)
	return util.Softmax(m.m.Predict(xs))
}

func (m *Model) Embeddings() (embs [][embDim]float64) {
	for idx := range m.vocabSize {
		v := gonet.NodeValues(m.emb.E(idx))
		embs = append(embs, [embDim]float64(v))
	}
	return embs
}

func (m *Model) Train(samples []util.Sample, cfg util.TrainConfig) time.Duration {
	var (
		totalLossFn = gonet.PredictLossFunc(m.m, gonet.CrossEntropyLoss)
		lossFn      = gonet.TrainLossFunc(m.m, gonet.CrossEntropyLoss)

		params    = m.m.Parameters()
		optimizer = util.DefaultAdamOptimizer(params, cfg.LearningRate)
		loss      *gonet.Node
		delta     float64
	)
	fmt.Println("Number of parameters:", len(params))

	// Evaluation before training.
	totalLoss := totalLossFn(samples)
	log.Printf("[Before training] total loss: %g", totalLoss)

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

		if ep%50 == 0 || ep+1 == cfg.Epochs {
			log.Printf("[Epoch=%d] Gradient descent delta=%g | loss=%g", ep, delta, loss.V())
		}
	}
	timeCost := time.Since(start)

	// Evaluation after training.
	totalLoss = totalLossFn(samples)
	log.Printf("[After training] total loss: %g", totalLoss)
	return timeCost
}

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

	inputs, labels := util.GenInputsAndLabelsFromCorpus(corpus, c2i, ctxLen)
	samples := util.GenDatasetFromInputsAndLabels(inputs, labels)

	model := newModel(vocabSize)
	for i := range 20 {
		name := makemore.GenName(i2c, model.PredictNextProbs, ctxLen)
		fmt.Printf("[Before training] Generate name (%d | len=%d): %s\n", i+1, len(name), name)
	}

	var (
		reader = bufio.NewReader(os.Stdin)
		cfg    = util.TrainConfig{
			BatchSize:    32,
			Epochs:       100,
			StopEps:      1e-8,
			LearningRate: 0.003,
		}
	)
training:
	for {
		fmt.Printf("Mini-batch size: %d\nTraining epochs: %d\nLearning rate: %g\n", cfg.BatchSize, cfg.Epochs, cfg.LearningRate)
		timeCost := model.Train(samples, cfg)
		log.Printf("Training time cost: %s", timeCost)

		if !*interactive {
			break
		}

		fmt.Println("Continue training?")
		fmt.Println("  (q, quit, exit) exit;")
		fmt.Println("  (integer float) training epochs -> integer, learning_rate -> float;")
		fmt.Println("  (otherwise) continue.")
		cmd, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalf("Read input command error: %v", err)
		}

		switch cmd = strings.TrimSpace(cmd); cmd {
		case "q", "quit", "exit":
			break training
		default:
			// Input command format:
			// (Case 1) 200
			// - Change training epochs to 200
			// (Case 2) 500 0.05
			// - Change training epochs to 500 and learning rate to 0.05
			fields := strings.Fields(cmd)
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
	printEmbeddings(model.Embeddings())

	// pnext = makemore.LazyProbNext(model, vocabSize, ctxLen)
	for i := range 20 {
		next := makemore.GenName(i2c, model.PredictNextProbs, ctxLen)
		fmt.Printf("[After training] Generate name (%d | len=%d): %s\n", i+1, len(next), next)
	}
}

func printEmbeddings(embs [][embDim]float64) {
	fmt.Println("Embeddings: [")
	for _, emb := range embs {
		fmt.Print("    [")
		for i, x := range emb {
			if i > 0 {
				fmt.Print(", ")
			}
			fmt.Printf("%.4f", x)
		}
		fmt.Printf("],\n")
	}
	fmt.Println("]")
}
