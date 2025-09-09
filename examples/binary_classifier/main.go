package main

import (
	"flag"

	"github.com/Lucas-884e/gonet/nnet"
)

var (
	data = flag.String("i", "data/data.csv", "Input data file name")
)

func f() {
	net := nnet.NewFCNNet(2, nnet.LogisticActivator(1), nnet.DLogisticActivator(1))
	net.AddLayer(2)
	net.AddLayer(3)
	net.AddLayer(2)

	net.RandomizeInitialWeights()
	net.PropagateSample([]float64{0.3, 0.7}, []float64{0.9, 0.1})
	net.Print()
}
