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

var data = flag.String("i", "../dataset/names.txt", "Input data file path")

const (
	ctxLen  = 3  // context length
	embDim  = 2  // embedding space dimension
	hidSize = 10 // hidden layer size
)

type Model struct {
	vocabSize int

	embeddingLayer *gonet.MLP
	predictionMLP  *gonet.MLP

	input gonet.SampleBatch
	loss  *gonet.Node
}

func newModel(vocabSize, sampleCount int) *Model {
	embedding := gonet.NewMLP(vocabSize)
	embedding.AddLayer(embDim, gonet.OpNone, false)

	predictor := gonet.NewMLP(embDim * ctxLen)
	predictor.AddLayer(hidSize, gonet.OpTanh, true)
	predictor.AddLayer(vocabSize, gonet.OpSoftmax, true)

	m := &Model{
		vocabSize:      vocabSize,
		embeddingLayer: embedding,
		predictionMLP:  predictor,
	}
	m.input, m.loss = buildLoss(m, vocabSize, sampleCount)
	return m
}

func (m *Model) Parameters() []util.Parameter {
	return append(m.embeddingLayer.Parameters(), m.predictionMLP.Parameters()...)
}

func (m *Model) Feed(input []*gonet.Node) []*gonet.Node {
	return m.predictionMLP.Feed(m.feedEmbeddingLayer(input))
}

func (m *Model) LazyOutput(input []*gonet.Node) func() []float64 {
	return m.predictionMLP.LazyOutput(m.feedEmbeddingLayer(input))
}

func (m *Model) feedEmbeddingLayer(input []*gonet.Node) []*gonet.Node {
	ctxEmbedding := make([]*gonet.Node, 0, ctxLen*embDim) // context embedding
	for i := 0; i < m.vocabSize*ctxLen; i += m.vocabSize {
		ohe := input[i : i+m.vocabSize]
		embedding := m.embeddingLayer.Feed(ohe)
		ctxEmbedding = append(ctxEmbedding, embedding...)
	}
	return ctxEmbedding
}

func (m *Model) Embeddings() (embs [][embDim]float64) {
	input := gonet.NewInputNodeBatch(m.vocabSize, "X_%d")
	output := m.embeddingLayer.Feed(input)
	for idx := range m.vocabSize {
		ohe := util.OneHot(idx, m.vocabSize)
		for i, in := range input {
			in.SetV(ohe[i])
		}
		for _, out := range output {
			out.Forward()
		}
		embs = append(embs, [embDim]float64(gonet.NodeValues(output)))
	}
	return embs
}

func (m *Model) Train(samples []util.Sample, cfg util.TrainConfig) {
	var (
		vocabSize             = len(samples[0].Y)
		batchInput, batchLoss = buildLoss(m, vocabSize, cfg.BatchSize)

		params    = m.Parameters()
		optimizer = util.DefaultAdamOptimizer(params, cfg.LearningRate)
		// optimizer = util.SGDOptimizer(params, cfg.LearningRate)
		delta float64
	)
	fmt.Println("Number of parameters:", len(params))

	// Evaluation before training.
	m.input.Update(samples)
	m.loss.Forward()
	log.Printf("[Before training] total loss: %g", m.loss.V())

	start := time.Now()
	fmt.Printf("Training epochs: %d\nLearning rate: %g\n", cfg.Epochs, cfg.LearningRate)
	for ep := 0; ep < cfg.Epochs; ep++ {
		for start := 0; start < len(samples); start += cfg.BatchSize {
			end := min(start+cfg.BatchSize, len(samples))
			batchInput.Update(samples[start:end])
			batchLoss.Backward()
			delta = optimizer.Learn()
		}

		if ep%50 == 0 || ep+1 == cfg.Epochs {
			log.Printf("[Epoch=%d] Gradient descent delta=%g | loss=%g", ep, delta, batchLoss.V())
		}
	}
	log.Printf("Training time cost: %s", time.Since(start))

	// Evaluation after training.
	m.loss.Forward()
	log.Printf("[After training] total loss: %g", m.loss.V())
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
	// Don't do evaluation on the entire dataset, it will eat all your memory.
	corpus = corpus[:500]
	log.Printf("Corpus size: %d", len(corpus))

	c2i := util.GenVocabFromCorpus(corpus, '.')
	i2c := util.GetIndexToToken(c2i)
	vocabSize := len(c2i)
	log.Printf("Vocabulary (size=%d): %c", vocabSize, i2c)

	inputs, labels := util.GenInputsAndLabelsFromCorpus(corpus, c2i, ctxLen)
	samples := util.GenDatasetFromInputsAndLabels(inputs, labels, ctxLen, vocabSize)

	var (
		model = newModel(vocabSize, len(samples))
		pnext = makemore.LazyProbNext(model, vocabSize, ctxLen)
	)
	for i := range 20 {
		fmt.Printf("[Before training] Generate name (%d): %s\n", i+1, makemore.GenName(i2c, pnext, ctxLen))
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
		model.Train(samples, cfg)

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

	pnext = makemore.LazyProbNext(model, vocabSize, ctxLen)
	for i := range 20 {
		next := makemore.GenName(i2c, pnext, ctxLen)
		fmt.Printf("[After training] Generate name (%d | len=%d): %s\n", i+1, len(next), next)
	}
}

func buildLoss(model *Model, vocabSize, inputSize int) (input gonet.SampleBatch, loss *gonet.Node) {
	input = gonet.NewSampleBatch(vocabSize*ctxLen, vocabSize, inputSize)
	loss = gonet.ModelLossFunc(model, gonet.CrossEntropyLoss)(input)
	return
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
