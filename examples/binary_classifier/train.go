package main

import (
	"encoding/csv"
	"flag"
	"fmt"
	"log"
	"os"
	"strconv"

	"github.com/Lucas-884e/gonet/nnet"
	"github.com/Lucas-884e/gonet/training"
)

type trainingSample struct {
	xs []float64
	ys []float64 // 0 -> class A, 1 -> class B
}

func main() {
	flag.Parse()

	records, err := readInRecords(*data)
	if err != nil {
		log.Fatal(err)
	}

	samples, err := recordsToTrainingSamples(records)
	if err != nil {
		log.Fatal(err)
	}
	normalizeTrainingSamplesByRemovingMeans(samples)
	training.ShuffleSamples(samples)
	trainingSet, validationSet, testingSet := training.SplitDataSet(samples)

	nn := constructNetwork()
	log.Printf("(Before training) Prediction precision: validation set = %g | testing set = %g",
		predictionPrecision(nn, validationSet),
		predictionPrecision(nn, testingSet))
	tsSize := len(trainingSet)
train:
	for epoch := 0; epoch < 50; epoch++ {
		// Shuffle before each epoch.
		training.ShuffleSamples(trainingSet)
		// One training epoch.
		for t, sample := range trainingSet {
			nn.PropagateSample(sample.xs, sample.ys)
			if nn.UpdateWeights(training.AnnealingLearningRate(0.1, 1000, int64(epoch*tsSize+t))) {
				log.Println("* Reached stopping criterion.")
				break train
			}
		}
		precision := predictionPrecision(nn, validationSet)
		log.Printf("[Epoch %d] Prediction precision: %g", epoch, precision)
	}
	nn.Print()

	log.Println("(After training) Prediction precision:", predictionPrecision(nn, testingSet))
}

func predictionPrecision(nn *nnet.FCNNet, dataSet []trainingSample) float32 {
	var correctCount int
	for _, sample := range dataSet {
		pred := nn.Predict(sample.xs)
		if pred[0]*sample.ys[0] > 0 {
			correctCount++
		}
	}
	return float32(correctCount) / float32(len(dataSet))
}

func constructNetwork() *nnet.FCNNet {
	nn := nnet.NewFCNNet(2, nnet.TanhActivator(1, 1), nnet.DTanhActivator(1, 1))
	nn.AddLayer(20)
	nn.AddLayer(1)
	nn.RandomizeInitialWeights()
	return nn
}

func normalizeTrainingSamplesByRemovingMeans(samples []trainingSample) (means []float64) {
	for _, sample := range samples {
		if len(means) == 0 {
			means = make([]float64, len(sample.xs))
		}
		for i, x := range sample.xs {
			means[i] += x
		}
	}
	total := float64(len(samples))
	for i := range means {
		means[i] /= total
	}

	for i := range samples {
		for j, mean := range means {
			samples[i].xs[j] -= mean
		}
	}
	return means
}

func recordsToTrainingSamples(records [][]string) (samples []trainingSample, err error) {
	for _, record := range records {
		xa, ya, xb, yb, err := recordToFloats(record)
		if err != nil {
			return nil, fmt.Errorf("Convert record %v: %v", record, err)
		}
		samples = append(samples, trainingSample{
			xs: []float64{xa, ya},
			ys: []float64{0.9},
		})
		samples = append(samples, trainingSample{
			xs: []float64{xb, yb},
			ys: []float64{-0.9},
		})
	}
	return samples, nil
}

func recordToFloats(record []string) (xa, ya, xb, yb float64, err error) {
	floats := []*float64{&xa, &ya, &xb, &yb}
	for i := 0; i < 4; i++ {
		*floats[i], err = strconv.ParseFloat(record[i], 64)
		if err != nil {
			return 0, 0, 0, 0, err
		}
	}
	return
}

func readInRecords(datafile string) ([][]string, error) {
	f, err := os.Open(*data)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	return r.ReadAll()
}
