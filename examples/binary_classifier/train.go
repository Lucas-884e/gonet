package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"

	"github.com/Lucas-884e/gonet"
)

var (
	data = flag.String("i", "data/data.csv", "Input data file name")
)

func main() {
	flag.Parse()

	records, err := gonet.ReadCSVDataSet(*data)
	if err != nil {
		log.Fatal(err)
	}

	samples, err := recordsToTrainingSamples(records)
	if err != nil {
		log.Fatal(err)
	}
	normalizeTrainingSamplesByRemovingMeans(samples)
	gonet.ShuffleSamples(samples)
	trainingSet, validationSet, testSet := gonet.SplitDataSet(samples)

	nn := constructNetwork(8)
	tr := gonet.NewTrainer(nn, func(pred, actual []float64) bool {
		return pred[0]*actual[0] > 0
	})
	log.Printf("(Before training) Prediction precision: validation set = %g | test set = %g",
		tr.PredictionPrecision(validationSet), tr.PredictionPrecision(testSet))

	tr.Train(gonet.TrainConfig{
		BatchSize:    1,
		Epochs:       50,
		StopEps:      0,
		LearningRate: 0.1,
	}, trainingSet, validationSet)

	nn.Print()
	log.Println("(After training) Test set prediction precision:", tr.PredictionPrecision(testSet))
}

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

func normalizeTrainingSamplesByRemovingMeans(samples []gonet.Sample) (means []float64) {
	for _, sample := range samples {
		if len(means) == 0 {
			means = make([]float64, len(sample.X))
		}
		for i, x := range sample.X {
			means[i] += x
		}
	}
	total := float64(len(samples))
	for i := range means {
		means[i] /= total
	}

	for i := range samples {
		for j, mean := range means {
			samples[i].X[j] -= mean
		}
	}
	return means
}

func recordsToTrainingSamples(records [][]string) (samples []gonet.Sample, err error) {
	for _, record := range records {
		x1a, x2a, x1b, x2b, err := recordToFloats(record)
		if err != nil {
			return nil, fmt.Errorf("Convert record %v: %v", record, err)
		}
		samples = append(samples, gonet.Sample{
			X: []float64{x1a, x2a},
			Y: []float64{1}, // Sample A: Class A
		})
		samples = append(samples, gonet.Sample{
			X: []float64{x1b, x2b},
			Y: []float64{-1}, // Sample B: Class B
		})
	}
	return samples, nil
}

// (x1a, x2a): coordinates with respect to (x1, x2) axes for sample A (of class A)
// (x1b, x2b): coordinates with respect to (x1, x2) axes for sample B (of class B)
func recordToFloats(record []string) (x1a, x2a, x1b, x2b float64, err error) {
	floats := []*float64{&x1a, &x2a, &x1b, &x2b}
	for i := 0; i < 4; i++ {
		*floats[i], err = strconv.ParseFloat(record[i], 64)
		if err != nil {
			return 0, 0, 0, 0, err
		}
	}
	return
}
