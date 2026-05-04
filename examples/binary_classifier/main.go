package main

import (
	"flag"
	"fmt"
	"log"
	"strconv"
	"time"

	"github.com/LucasInOz/gonet/util"
)

var (
	data     = flag.String("i", "data/data.csv", "Input data file name")
	useArray = flag.Bool("arr", false, "use array-based approach")
)

func main() {
	flag.Parse()

	records, err := util.ReadCSVDataSet(*data)
	if err != nil {
		log.Fatal(err)
	}

	samples, err := recordsToTrainingSamples(records)
	if err != nil {
		log.Fatal(err)
	}
	normalizeTrainingSamplesByRemovingMeans(samples)
	util.ShuffleSamples(samples)

	trainingSet, validationSet, _ := util.SplitDataSet(samples, 0.8, 1.0)
	train := graphTrain
	if *useArray {
		train = nonGraphTrain
	}

	start := time.Now()
	train(trainingSet, validationSet)
	fmt.Println("\nTraining time cost:", time.Since(start))
}

func normalizeTrainingSamplesByRemovingMeans(samples []util.Sample) (means []float64) {
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

func recordsToTrainingSamples(records [][]string) (samples []util.Sample, err error) {
	for _, record := range records {
		x1a, x2a, x1b, x2b, err := recordToFloats(record)
		if err != nil {
			return nil, fmt.Errorf("convert record %v: %v", record, err)
		}
		samples = append(samples, util.Sample{
			X: []float64{x1a, x2a},
			Y: []float64{1}, // Sample A: Class A
		})
		samples = append(samples, util.Sample{
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
	for i, f := range floats {
		*f, err = strconv.ParseFloat(record[i], 64)
		if err != nil {
			return 0, 0, 0, 0, err
		}
	}
	return
}
