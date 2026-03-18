package main

import (
	"fmt"
	"log"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/graph"
	"github.com/Lucas-884e/gonet/util"
)

func constructNonGraphNetwork(inputLayerSize int, hiddenLayerSizes ...int) *gonet.FCNNet {
	nn := gonet.NewFCNNet(inputLayerSize, gonet.LossCrossEntropy, gonet.ReluActivator())
	for _, size := range hiddenLayerSizes {
		nn.AddLayer(size)
	}
	nn.AddLayerWithActivator(10, gonet.SoftmaxActivator(1))
	nn.RandomizeInitialWeights()
	return nn
}

func nonGraphTrain(trainingSet, validationSet, testSet []util.Sample) {
	nn := constructNonGraphNetwork(len(trainingSet[0].X), 32, 16)
	tr := gonet.NewTrainer(nn, isCorrect)
	log.Printf("(Before training) Prediction precision: validation set = %g | testing set = %g",
		tr.PredictionPrecision(validationSet), tr.PredictionPrecision(testSet))

	tr.Train(util.TrainConfig{
		BatchSize:    10,
		Epochs:       20,
		StopEps:      0,
		LearningRate: 0.05,
	}, trainingSet, validationSet)

	// nn.Print()
	log.Println("(After training) Testing set prediction precision:", tr.PredictionPrecision(testSet))
}

func constructGraphNetwork(inputLayerSize int, hiddenLayerSizes ...int) *graph.MLP {
	mlp := graph.NewMLP(inputLayerSize)
	for _, size := range hiddenLayerSizes {
		mlp.AddLayer(size, graph.OpRelu, true)
	}
	mlp.AddLayer(10, graph.OpSoftmax, true)
	return mlp
}

func graphTrain(trainingSet, validationSet, testSet []util.Sample) {
	var (
		inputLayerSize = len(trainingSet[0].X)
		mlp            = constructGraphNetwork(inputLayerSize, 32, 16)
		lossFn         = graph.ModelLossFunc(mlp, graph.CrossEntropyLoss)
		tsSize         = len(trainingSet)
		delta          float64
	)

	precision := PredictionPrecision(mlp, validationSet)
	log.Printf("[Before training] Validation set prediction precision: %g", precision)
	// fmt.Println(mlp)

	var (
		cfg = util.TrainConfig{
			BatchSize:    10,
			Epochs:       20,
			StopEps:      0,
			LearningRate: 0.05,
		}
		batchInput = graph.NewSampleBatch(inputLayerSize, 10, cfg.BatchSize)
		loss       = lossFn(batchInput)
	)
train:
	for ep := 0; ep < cfg.Epochs; ep++ {
		// Shuffle before each epoch.
		util.ShuffleSamples(trainingSet)
		for start := 0; start < tsSize; start += cfg.BatchSize {
			end := min(start+cfg.BatchSize, tsSize)
			batchInput.Update(trainingSet[start:end])
			loss.Backward()

			lrFunc := util.AnnealingLearningRateFunc(cfg.LearningRate, 1, ep)
			if delta = mlp.Learn(lrFunc); delta < cfg.StopEps {
				log.Printf("* Reached stopping criterion (delta = %g < %g).", delta, cfg.StopEps)
				break train
			}
		}

		if ep%5 == 0 || ep+1 == cfg.Epochs {
			precision := PredictionPrecision(mlp, validationSet)
			log.Printf("[Epoch %d] Validation set prediction precision (delta=%g): %g", ep+1, delta, precision)
		}
		// fmt.Println(mlp)
	}

	precision = PredictionPrecision(mlp, testSet)
	log.Printf("(After training) Test set prediction precision: %g", precision)
	// fmt.Println(mlp)
}

func PredictionPrecision(model *graph.MLP, dataset []util.Sample) float32 {
	var (
		correctCount int
		input        = graph.NewInputNodeBatch(len(dataset[0].X), "X_%d")
		predicted    = model.Feed(input)
	)
	for _, sample := range dataset {
		for i, x := range sample.X {
			input[i].SetV(x)
		}
		for _, pred := range predicted {
			pred.Forward()
		}
		if isCorrect(graph.NodeValues(predicted), sample.Y) {
			correctCount++
		}
	}
	fmt.Println("Correct count:", correctCount, "| Total count:", len(dataset))
	return float32(correctCount) / float32(len(dataset))
}

func isCorrect(pred, actual []float64) bool {
	findPrediction := func(ys []float64) int {
		maxIdx := -1
		var maxProb float64
		for i, y := range ys {
			if y > maxProb {
				maxProb = y
				maxIdx = i
			}
		}
		return maxIdx
	}
	return findPrediction(pred) == findPrediction(actual)
}
