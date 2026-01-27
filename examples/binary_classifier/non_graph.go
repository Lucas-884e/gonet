package main

import (
	"log"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/util"
)

func constructNetwork(hiddenLayerSizes ...int) *gonet.FCNNet {
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
	nn := constructNetwork(8)
	tr := gonet.NewTrainer(nn, func(pred, actual []float64) bool {
		return pred[0]*actual[0] > 0
	})
	log.Printf("(Before training) Prediction precision: validation set = %g | test set = %g",
		tr.PredictionPrecision(validationSet), tr.PredictionPrecision(testSet))

	tr.Train(util.TrainConfig{
		BatchSize:    1,
		Epochs:       20,
		StopEps:      0,
		LearningRate: 0.01,
	}, trainingSet, validationSet)

	nn.Print()
	log.Println("(After training) Test set prediction precision:", tr.PredictionPrecision(testSet))
}
