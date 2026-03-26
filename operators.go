package gonet

//go:generate stringer -type=Operator
type Operator int32

const (
	OpNone Operator = iota
	OpPlus
	OpMultiply
	OpRelu
	OpSigmoid
	OpTanh
	OpSoftmax
)
