package main

import (
	"log"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/graph"
	"github.com/Lucas-884e/gonet/util"
)

func constructNonGraphNetwork(hiddenLayerSizes ...int) *gonet.FCNNet {
	// Must use Tanh activator because the training data has target values within range: [-1, 1]
	nn := gonet.NewFCNNet(2, gonet.LossMaxMargin, gonet.ReluActivator())
	for _, size := range hiddenLayerSizes {
		nn.AddLayer(size)
	}
	nn.AddLayerWithActivator(1, gonet.LinearActivator())
	nn.RandomizeInitialWeights()
	return nn
}

func nonGraphTrain(trainingSet, validationSet, testSet []util.Sample) {
	nn := constructNonGraphNetwork(8)
	tr := gonet.NewTrainer(nn, func(pred, actual []float64) bool {
		return pred[0]*actual[0] > 0
	})
	log.Printf("(Before training) Prediction precision: validation set = %g | test set = %g",
		tr.PredictionPrecision(validationSet), tr.PredictionPrecision(testSet))

	tr.Train(util.TrainConfig{
		BatchSize:    20,
		Epochs:       20,
		StopEps:      0,
		LearningRate: 0.3,
	}, trainingSet, validationSet)

	nn.Print()
	log.Println("(After training) Test set prediction precision:", tr.PredictionPrecision(testSet))
}

func constructGraphNetwork(hiddenLayerSizes ...int) *graph.MLP {
	mlp := graph.NewMLP(2)
	for _, size := range hiddenLayerSizes {
		mlp.AddLayer(size, graph.OpRelu, true)
	}
	mlp.AddLayer(1, graph.OpNone, true)
	return mlp
}

func graphTrain(trainingSet, validationSet, testSet []util.Sample) {
	var (
		mlp       = constructGraphNetwork(8)
		lossFn    = graph.ModelLossFunc(mlp, graph.MaxMarginLoss)
		isCorrect = func(pred, actual []float64) bool { return pred[0]*actual[0] > 0 }
		tsSize    = len(trainingSet)
		delta     float64
	)

	precision := PredictionPrecision(mlp, validationSet, isCorrect)
	log.Printf("[Before training] Validation set prediction precision: %g", precision)

	var (
		cfg = util.TrainConfig{
			BatchSize:    20,
			Epochs:       20,
			StopEps:      0,
			LearningRate: 0.3,
		}
		batchInput = graph.NewSampleBatch(2, 1, cfg.BatchSize)
		loss       = lossFn(batchInput)
		optimizer  = util.NewDefaultAdamOptimizer(cfg.LearningRate)
	)
train:
	for ep := 0; ep < cfg.Epochs; ep++ {
		// Shuffle before each epoch.
		util.ShuffleSamples(trainingSet)
		for start := 0; start < tsSize; start += cfg.BatchSize {
			end := min(start+cfg.BatchSize, tsSize)
			batchInput.Update(trainingSet[start:end])
			loss.Backward()

			optimizer.Step()
			if delta = mlp.Learn(optimizer.LearningRate); delta < cfg.StopEps {
				log.Printf("* Reached stopping criterion (delta = %g < %g).", delta, cfg.StopEps)
				break train
			}
		}

		if ep%5 == 0 || ep+1 == cfg.Epochs {
			precision := PredictionPrecision(mlp, validationSet, isCorrect)
			log.Printf("[Epoch %d] Validation set prediction precision (delta=%g): %g", ep+1, delta, precision)
		}
	}

	precision = PredictionPrecision(mlp, testSet, isCorrect)
	log.Printf("[After training] Test set prediction precision: %g", precision)
	// fmt.Println(mlp)
}

func PredictionPrecision(model *graph.MLP, dataset []util.Sample, isCorrect util.IsCorrectFunc) float32 {
	var (
		correctCount int
		input        = []*graph.Node{
			graph.NewInputNode(0, "X1"),
			graph.NewInputNode(0, "X2"),
		}
		predicted = model.Feed(input)
	)
	for _, sample := range dataset {
		for i, x := range sample.X {
			input[i].SetV(x)
		}
		predicted[0].Forward()
		if isCorrect(graph.NodeValues(predicted), sample.Y) {
			correctCount++
		}
	}
	return float32(correctCount) / float32(len(dataset))
}
