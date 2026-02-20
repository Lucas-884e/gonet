package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"

	"github.com/Lucas-884e/gonet"
	"github.com/Lucas-884e/gonet/util"
)

var dataset = flag.String("ds", "sklearn", "Dataset name.")

func main() {
	flag.Parse()

	var (
		trainingSet   []util.Sample
		validationSet []util.Sample
		testSet       []util.Sample
	)

	switch *dataset {
	case "sklearn":
		records, err := util.ReadCSVDataSet("data/sklearn/digits.csv")
		if err != nil {
			log.Fatal(err)
		}
		samples, err := recordsToTrainingSamples(records)
		if err != nil {
			log.Fatal(err)
		}

		util.ShuffleSamples(samples)
		trainingSet, validationSet, testSet = util.SplitDataSet(samples)

	case "mnist":
		trainingRecords, err := util.ReadCSVDataSet("data/mnist/training_data.csv")
		if err != nil {
			log.Fatal(err)
		}
		trainingSet, err = recordsToTrainingSamples(trainingRecords)
		if err != nil {
			log.Fatal(err)
		}

		validationRecords, err := util.ReadCSVDataSet("data/mnist/validation_data.csv")
		if err != nil {
			log.Fatal(err)
		}
		validationSet, err = recordsToTrainingSamples(validationRecords)
		if err != nil {
			log.Fatal(err)
		}

		testRecords, err := util.ReadCSVDataSet("data/mnist/test_data.csv")
		if err != nil {
			log.Fatal(err)
		}
		testSet, err = recordsToTrainingSamples(testRecords)
		if err != nil {
			log.Fatal(err)
		}
	}

	nn := constructNetwork(len(trainingSet[0].X), 32, 16)
	tr := gonet.NewTrainer(nn, isCorrect)
	log.Printf("(Before training) Prediction precision: validation set = %g | testing set = %g",
		tr.PredictionPrecision(validationSet), tr.PredictionPrecision(testSet))

	tr.Train(util.TrainConfig{
		BatchSize:    1,
		Epochs:       30,
		StopEps:      0,
		LearningRate: 0.01,
	}, trainingSet, validationSet)

	// nn.Print()
	log.Println("(After training) Testing set prediction precision:", tr.PredictionPrecision(testSet))
}

func constructNetwork(inputLayerSize int, hiddenLayerSizes ...int) *gonet.FCNNet {
	nn := gonet.NewFCNNet(inputLayerSize, gonet.LossCrossEntropy, gonet.ReluActivator())
	for _, size := range hiddenLayerSizes {
		nn.AddLayer(size)
	}
	nn.AddLayerWithActivator(10, gonet.SoftmaxActivator(1))
	nn.RandomizeInitialWeights()
	return nn
}

func isCorrect(pred, actual []float64) bool {
	findPrediction := func(ys []float64) int {
		maxIdx := -1
		var maxProb float64
		for i, y := range ys {
			if y > maxProb {
				maxProb = y
				maxIdx = i
			}
		}
		return maxIdx
	}
	return findPrediction(pred) == findPrediction(actual)
}

func recordsToTrainingSamples(records [][]string) (samples []util.Sample, err error) {
	for i, record := range records {
		X, Y, err := recordToFloats(record)
		if err != nil {
			return nil, fmt.Errorf("convert %d-th sample %v: %v", i, record, err)
		}
		samples = append(samples, util.Sample{X: X, Y: Y})
	}
	return samples, nil
}

// record = [x1, x2, ..., xN, y]
func recordToFloats(record []string) (X, Y []float64, err error) {
	vecLen := len(record) - 1

	X = make([]float64, vecLen)
	for i := range X {
		X[i], err = strconv.ParseFloat(record[i], 64)
		if err != nil {
			return nil, nil, fmt.Errorf("read %d-th element of input vector (%s): %w", i, record[i], err)
		}
	}

	Y = make([]float64, 10)
	d, err := strconv.ParseFloat(record[vecLen], 64)
	if err != nil {
		return X, nil, fmt.Errorf("read ground truth digit (%s): %w", record[vecLen], err)
	}
	if d < 0 || d > 9 {
		return X, nil, fmt.Errorf("invalid ground truth digit: %g", d)
	}
	Y[int(d)] = 1.0

	return X, Y, nil
}
