package graph

import (
	"testing"

	"github.com/stretchr/testify/assert"
)

func TestNode(t *testing.T) {
	var (
		x1 = NewNode(-5, "x1")
		x2 = NewNode(2, "x2")
	)
	assert.Equal(t, -5.0, x1.V())
	assert.Equal(t, "x1", x1.Name())

	var (
		b1  = NewNode(1, "b1") // bias
		w11 = NewNode(0.5, "w11")
		w12 = NewNode(0, "w12")

		b2  = NewNode(-0.5, "b2") // bias
		w21 = NewNode(-1, "w21")
		w22 = NewNode(0, "w22")

		ksi11 = Multiply(w11, x1)      // -2.5
		ksi12 = Multiply(w12, x2)      // 0
		ksi1  = Plus(b1, ksi11, ksi12) // -1.5
		y1    = Sigmoid(ksi1)

		ksi21 = Multiply(w21, x1)      // 5
		ksi22 = Multiply(w22, x2)      // 0
		ksi2  = Plus(b2, ksi21, ksi22) // 4.5
		y2    = Sigmoid(ksi2)
	)
	assert.Equal(t, -2.5, ksi11.V())
	assert.Equal(t, "w11×x1", ksi11.Name())
	assert.Equal(t, -1.5, ksi1.V())
	assert.Equal(t, "b1+w11×x1+w12×x2", ksi1.Name())
	assert.Equal(t, 0.18242552380635635, y1.V())
	assert.Equal(t, "σ(b1+w11×x1+w12×x2)", y1.Name())
	assert.Equal(t, 0.9890130573694068, y2.V())
	assert.Equal(t, "σ(b2+w21×x1+w22×x2)", y2.Name())

	var (
		c1  = NewNode(0, "c1") // bias
		u11 = NewNode(-0.5, "u11")
		u12 = NewNode(1, "u12")

		c2  = NewNode(-1, "c2") // bias
		u21 = NewNode(1, "u21")
		u22 = NewNode(2, "u22")

		eta11 = Multiply(u11, y1)
		eta12 = Multiply(u12, y2)
		eta1  = Plus(c1, eta11, eta12)

		eta21 = Multiply(u21, y1)
		eta22 = Multiply(u22, y2)
		eta2  = Plus(c2, eta21, eta22)
	)
	assert.Equal(t, 0.8978002954662286, eta1.V())
	assert.Equal(t, "c1+u11×σ(b1+w11×x1+w12×x2)+u12×σ(b2+w21×x1+w22×x2)", eta1.Name())
	assert.Equal(t, 1.16045163854517, eta2.V())
	assert.Equal(t, "c2+u21×σ(b1+w11×x1+w12×x2)+u22×σ(b2+w21×x1+w22×x2)", eta2.Name())

	var (
		zs = Softmax(1, eta1, eta2)
		z1 = zs[0]
		z2 = zs[1]
	)
	assert.Equal(t, 0.43471206139788876, z1.V())
	assert.Equal(t, "softmax[0](1)", z1.Name())
	assert.Equal(t, 0.5652879386021112, z2.V())
	assert.Equal(t, "softmax[1](1)", z2.Name())

	ce := CrossEntropy(0, z1, z2) // The 0-th element is observed.
	ce.Backward()
}
