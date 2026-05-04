package main

import (
	"log"

	"github.com/LucasInOz/gonet"
	"github.com/LucasInOz/gonet/arrimpl"
	"github.com/LucasInOz/gonet/util"
)

func constructNonGraphNetwork(hiddenLayerSizes ...int) *arrimpl.MLP {
	// Must use Tanh activator because the training data has target values within range: [-1, 1]
	nn := arrimpl.NewMLP(2, arrimpl.LossMaxMargin, arrimpl.ReluActivator())
	for _, size := range hiddenLayerSizes {
		nn.AddLayer(size)
	}
	nn.AddLayerWithActivator(1, arrimpl.LinearActivator())
	nn.RandomizeInitialWeights()
	return nn
}

func nonGraphTrain(trainingSet, validationSet []util.Sample) {
	nn := constructNonGraphNetwork(8)
	tr := arrimpl.NewTrainer(nn, func(pred, actual []float64) bool {
		return pred[0]*actual[0] > 0
	})
	log.Printf("[Before training] validation set prediction precision: %g", tr.PredictionPrecision(validationSet))

	tr.Train(util.TrainConfig{
		BatchSize:    20,
		Epochs:       20,
		StopEps:      0,
		LearningRate: 0.3,
	}, trainingSet, validationSet)

	nn.Print()
	log.Println("[After training] Validation set prediction precision:", tr.PredictionPrecision(validationSet))
}

func constructGraphNetwork(hiddenLayerSizes ...int) gonet.Model {
	var (
		fanIn  = 2
		layers []gonet.Layer
	)
	for _, fanOut := range hiddenLayerSizes {
		layers = append(layers,
			gonet.LinearLayer(fanIn, fanOut, true),
			gonet.ReluLayer())
		fanIn = fanOut
	}
	layers = append(layers, gonet.LinearLayer(fanIn, 1, true))
	return gonet.SequentialModel(layers...)
}

func graphTrain(trainingSet, validationSet []util.Sample) {
	var (
		mlp       = constructGraphNetwork(8)
		isCorrect = func(pred, actual []float64) bool { return pred[0]*actual[0] > 0 }
		precision = util.PredictionPrecision(mlp, validationSet, isCorrect)
	)
	log.Printf("[Before training] Validation set prediction precision: %g", precision)

	var (
		cfg = util.TrainConfig{
			BatchSize:    20,
			Epochs:       20,
			StopEps:      0,
			LearningRate: 0.03,
		}
		timeCost = gonet.Train(mlp, trainingSet, &cfg, gonet.MaxMarginLoss)
	)
	log.Printf("Training time cost: %s", timeCost)

	precision = util.PredictionPrecision(mlp, validationSet, isCorrect)
	log.Printf("[After training] Validation set prediction precision: %g", precision)
}
