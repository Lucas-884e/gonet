package main

import (
	"github.com/Lucas-884e/gonet/nnet"
)

func predict(hiddenLayerSizes ...int) {
	net := nnet.NewFCNNet(2, nnet.TanhActivator(1, 1))
	for _, size := range hiddenLayerSizes {
		net.AddLayer(size)
	}
	net.AddLayer(1)

	net.RandomizeInitialWeights()
	net.PropagateSample([]float64{0.3, 0.7}, []float64{0.9, 0.1})
	net.Print()
}
