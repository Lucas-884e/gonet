package gonet

type LossFunction string

const (
	LossRSS          LossFunction = "Residual-Sum-of-Squared"
	LossMaxMargin    LossFunction = "Max-Margin"
	LossCrossEntropy LossFunction = "Cross-Entropy"
)

func (loss LossFunction) Grads(pred, actual []float64) []float64 {
	switch loss {
	case LossRSS:
		return sseGrads(pred, actual)
	case LossMaxMargin:
		return maxMarginGrads(pred, actual)
	case LossCrossEntropy:
		return crossEntropyGrads(pred, actual)
	default:
		panic("Unsupported Loss Function: " + loss)
	}
}

func sseGrads(pred, actual []float64) []float64 {
	g := make([]float64, len(pred))
	for i, p := range pred {
		g[i] = p - actual[i]
	}
	return g
}

func maxMarginGrads(pred, actual []float64) []float64 {
	g := make([]float64, len(pred))
	for i, d := range actual {
		if pred[i]*d < 1 {
			g[i] = -d
		}
	}
	return g
}

func crossEntropyGrads(predicted, observed []float64) []float64 {
	g := make([]float64, len(predicted))
	for i, ob := range observed {
		if ob > 0 {
			// ob is the observed probability, if it's not 0, it must be 1.
			g[i] = -ob / predicted[i]
		}
	}
	return g
}
