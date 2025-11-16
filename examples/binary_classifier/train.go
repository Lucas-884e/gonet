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

var (
	data = flag.String("i", "data/data.csv", "Input data file name")
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

	nn := constructNetwork(5)
	log.Printf("(Before training) Prediction precision: validation set = %g | testing set = %g",
		predictionPrecision(nn, validationSet),
		predictionPrecision(nn, testingSet))

	// Training process.
	tsSize := len(trainingSet)
train:
	for epoch := 0; epoch < 50; epoch++ {
		// Shuffle before each epoch.
		training.ShuffleSamples(trainingSet)
		// One training epoch.
		for t, sample := range trainingSet {
			nn.PropagateSample(sample.xs, sample.ys)

			learningRate := training.AnnealingLearningRate(0.1, 1000, int64(epoch*tsSize+t))
			if eps := nn.UpdateWeights(learningRate); eps < 1e-9 {
				log.Printf("* Reached stopping criterion (epsilon = %g).", eps)
				break train
			}
		}

		precision := predictionPrecision(nn, validationSet)
		log.Printf("[Epoch %d] Validation set prediction precision: %g", epoch+1, precision)
	}

	log.Println("(After training) Testing set prediction precision:", predictionPrecision(nn, testingSet))
	nn.Print()
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

func constructNetwork(hiddenLayerSizes ...int) *nnet.FCNNet {
	// Must use Tanh activator because the training data has target values within range: [-1, 1]
	nn := nnet.NewFCNNet(2, nnet.TanhActivator(1, 1))
	for _, size := range hiddenLayerSizes {
		nn.AddLayer(size)
	}
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
		x1a, x2a, x1b, x2b, err := recordToFloats(record)
		if err != nil {
			return nil, fmt.Errorf("Convert record %v: %v", record, err)
		}
		samples = append(samples, trainingSample{
			xs: []float64{x1a, x2a},
			ys: []float64{0.9}, // Sample A: Class A
		})
		samples = append(samples, trainingSample{
			xs: []float64{x1b, x2b},
			ys: []float64{-0.9}, // Sample B: Class B
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

func readInRecords(datafile string) ([][]string, error) {
	f, err := os.Open(*data)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	r := csv.NewReader(f)
	return r.ReadAll()
}
