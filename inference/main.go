package main

import (
	"github.com/Lucas-884e/gonet/nnet"
)

func main() {
	net := nnet.NewFCNNet(2, nnet.LogisticActivator(1), nnet.DLogisticActivator(1))
	net.AddLayer(2)
	net.AddLayer(3)
	net.AddLayer(2)

	net.RandomizeInitialWeights()
	net.PropagateSample([]float64{0.3, 0.7}, []float64{0.9, 0.1})
	net.Print()
}
