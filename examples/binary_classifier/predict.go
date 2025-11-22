package main

import (
	"github.com/Lucas-884e/gonet"
)

func predict(hiddenLayerSizes ...int) {
	net := gonet.NewFCNNet(2, gonet.LossMaxMargin, gonet.ReluActivator())
	for _, size := range hiddenLayerSizes {
		net.AddLayer(size)
	}
	net.AddLayerWithActivator(1, gonet.LinearActivator())

	net.RandomizeInitialWeights()
	net.Predict([]float64{0.3, 0.7})
	net.Print()
}
