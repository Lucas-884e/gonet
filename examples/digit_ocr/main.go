package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"
	"time"

	"github.com/Lucas-884e/gonet/util"
)

var (
	dataset  = flag.String("ds", "sklearn", "Dataset name.")
	useArray = flag.Bool("arr", false, "use array-based approach")
)

func main() {
	flag.Parse()

	var (
		trainingSet   []util.Sample
		validationSet []util.Sample
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
		trainingSet, validationSet, _ = util.SplitDataSet(samples, 0.8, 1.0)

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
	}

	train := graphTrain
	if *useArray {
		train = nonGraphTrain
	}

	start := time.Now()
	train(trainingSet, validationSet)
	fmt.Println("\nTraining time cost:", time.Since(start))
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
