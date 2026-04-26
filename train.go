package gonet

import (
	"cmp"
	"fmt"
	"log"
	"time"

	"github.com/Lucas-884e/gonet/util"
)

func Train(model Model, samples []util.Sample, cfg *util.TrainConfig, lf LossFunction) time.Duration {
	var (
		logInterval = cmp.Or(cfg.LogEpochInterval, 10)
		params      = model.Parameters()
		optimizer   = util.DefaultAdamOptimizer(params, cfg.LearningRate)
		totalLossFn = PredictLossFunc(model, lf)
		lossFn      = TrainLossFunc(model, lf)
		loss        *Node
		delta       float64
	)
	fmt.Println("Number of parameters:", len(params))

	// Evaluation before training.
	totalLoss := totalLossFn(samples)
	log.Printf("[Before training] total loss: %g", totalLoss)

	start := time.Now()
	for ep := 0; ep < cfg.Epochs; ep++ {
		util.ShuffleSamples(samples)
		for start := 0; start < len(samples); start += cfg.BatchSize {
			end := start + cfg.BatchSize
			if end > len(samples) {
				break // Ignore samples left that cannot form a mini-batch.
			}
			loss = lossFn(samples[start:end])
			loss.Backward()
			delta = optimizer.Learn()
		}

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
