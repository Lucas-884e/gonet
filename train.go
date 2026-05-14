package gonet

import (
	"cmp"
	"fmt"
	"log"
	"math"
	"time"

	"github.com/LucasInOz/gonet/util"
)

func Train(model Model, samples [][]util.Sample, cfg *util.TrainConfig, lf LossFunction) time.Duration {
	var (
		params    = model.Parameters()
		optimizer util.Optimizer
	)
	switch cfg.Optimizer {
	case "sgd":
		optimizer = util.SGDOptimizer(params, cfg.LearningRate)
	case "adam":
		fallthrough
	default:
		optimizer = util.DefaultAdamOptimizer(params, cfg.LearningRate)
	}
	fmt.Println("Number of parameters:", len(params))

	// Evaluation before training.
	var (
		evalLossFn = PredictLossFunc(model, lf)
		trsetLoss  = evalLossFn(samples[0])
		valsetLoss = math.NaN()
	)
	if len(samples) > 1 {
		valsetLoss = evalLossFn(samples[1])
	}
	log.Printf("[Before training] training set loss: %g | validation set loss: %g", trsetLoss, valsetLoss)

	var (
		delta        float64
		loss         *Node
		lossFn       = TrainLossFunc(model, lf)
		trainSamples = samples[0]
		logInterval  = cmp.Or(cfg.LogEpochInterval, 10)
		start        = time.Now()
	)
	for ep := 0; ep < cfg.Epochs; ep++ {
		// Prepare training samples for this epoch.
		if cfg.Sampler != nil {
			trainSamples = cfg.Sampler()
		}
		util.ShuffleSamples(trainSamples)

		// Do forward & backward propagation and update parameters.
		for start := 0; start < len(trainSamples); start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > len(trainSamples) {
				break // Ignore samples left that cannot form a mini-batch.
			}
			loss = lossFn(trainSamples[start:end])
			loss.Backward()
			delta = optimizer.Learn()
		}

		// Emit some metrics.
		if ep%logInterval == 0 || ep+1 == cfg.Epochs {
			log.Printf("[Epoch=%d] Gradient descent delta=%g | loss=%g", ep, delta, loss.V())
		}
	}
	timeCost := time.Since(start)

	// Evaluation after training.
	trsetLoss = evalLossFn(samples[0])
	if len(samples) > 1 {
		valsetLoss = evalLossFn(samples[1])
	}
	log.Printf("[After training] training set loss: %g | validation set loss: %g", trsetLoss, valsetLoss)
	return timeCost
}
