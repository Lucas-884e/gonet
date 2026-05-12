package util

import (
	"bufio"
	"fmt"
	"log"
	"os"
	"strconv"
	"strings"
	"time"
)

type TrainConfig struct {
	BatchSize        int
	Epochs           int
	StopEps          float64
	Optimizer        string
	LearningRate     float64
	LogEpochInterval int

	// If this function isn't nil, the trainer will use the samples returned from
	// this function for training in each epoch.
	Sampler func() []Sample
}

type IsCorrectFunc func(pred, label []float64) bool

type Predictor interface {
	Predict([]float64) []float64
}

func PredictionPrecision(p Predictor, testSet []Sample, isCorrect IsCorrectFunc) float32 {
	var correctCount int
	for _, sample := range testSet {
		if isCorrect(p.Predict(sample.X), sample.Y) {
			correctCount++
		}
	}
	return float32(correctCount) / float32(len(testSet))
}

func InteractiveTrain(cfg *TrainConfig, interactive bool, train func() time.Duration, predict func(...string)) {
	fmt.Printf("Mini-batch size: %d\nTraining epochs: %d\nLearning rate: %g\n", cfg.BatchSize, cfg.Epochs, cfg.LearningRate)
	trainTimeCost := train()
	log.Printf("Training time cost: %s", trainTimeCost)

	if !interactive {
		return
	}

training:
	for reader := bufio.NewReader(os.Stdin); ; {

		fmt.Println("Continue training?")
		fmt.Println("  (q) quit/exit;")
		fmt.Println("  (p) do prediction;")
		fmt.Println("  (integer float) training epochs -> integer, learning_rate -> float;")
		fmt.Println("  (otherwise) continue.")
		cmd, err := reader.ReadString('\n')
		if err != nil {
			log.Fatalf("Read input command error: %v", err)
		}

		switch fields := strings.Fields(strings.TrimSpace(cmd)); {
		case len(fields) == 1 && fields[0] == "q":
			break training

		case len(fields) >= 1 && fields[0] == "p":
			if predict != nil {
				predict(fields[1:]...)
			} else {
				log.Println("predict() function is not provided.")
			}

		default:
			// Input command format:
			// (Case 1) 200
			// - Change training epochs to 200
			// (Case 2) 500 0.05
			// - Change training epochs to 500 and learning rate to 0.05
			// fields := strings.Fields(cmd)
			if len(fields) > 0 {
				if ep, err := strconv.Atoi(fields[0]); err == nil {
					cfg.Epochs = ep
				}
			}
			if len(fields) > 1 {
				if lr, err := strconv.ParseFloat(fields[1], 64); err == nil {
					cfg.LearningRate = lr
				}
			}

			fmt.Println("Mini-batch size:", cfg.BatchSize)
			fmt.Println("Training epochs:", cfg.Epochs)
			fmt.Println("Learning rate:", cfg.LearningRate)
			trainTimeCost := train()
			log.Println("Training time cost:", trainTimeCost)
		}
	}
}
