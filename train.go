package gonet

import (
	"cmp"
	"fmt"
	"log"
	"time"

	"github.com/LucasInOz/gonet/util"
)

func Train(model Model, samples []util.Sample, cfg *util.TrainConfig, lf LossFunction) time.Duration {
	var (
		logInterval  = cmp.Or(cfg.LogEpochInterval, 10)
		params       = model.Parameters()
		totalLossFn  = PredictLossFunc(model, lf)
		lossFn       = TrainLossFunc(model, lf)
		trainSamples = samples
		loss         *Node
		optimizer    util.Optimizer
		delta        float64
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
	totalLoss := totalLossFn(samples)
	log.Printf("[Before training] total loss: %g", totalLoss)

	start := time.Now()
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
	totalLoss = totalLossFn(samples)
	log.Printf("[After training] total loss: %g", totalLoss)
	return timeCost
}
