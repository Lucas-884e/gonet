package main

import (
	"encoding/csv"
	"flag"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"time"
)

var (
	output   = flag.String("o", "data.csv", "Output csv file name")
	count    = flag.Int("c", 10, "Sample count")
	radius   = flag.Float64("r", 10, "Radius")
	width    = flag.Float64("w", 6, "Width")
	distance = flag.Float64("d", -4, "Distance")
)

func randR(r, w float64) float64 {
	return r + (rand.Float64()-0.5)*w
}

func randTheta() float64 {
	return math.Pi * rand.Float64()
}

func randClassAB(r, w, d float64) (xA, yA, xB, yB float64) {
	R := randR(r, w)
	theta := randTheta()
	xA, yA = R*math.Cos(theta), R*math.Sin(theta)
	xB, yB = r+xA, -d-yA
	return xA, yA, xB, yB
}

func toStrings(vs []float64) (ss []string) {
	for _, v := range vs {
		ss = append(ss, strconv.FormatFloat(v, 'f', -1, 64))
	}
	return ss
}

func main() {
	flag.Parse()

	f, err := os.Create(*output)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()
	w := csv.NewWriter(f)

	rand.Seed(time.Now().Unix())
	for i := 0; i < *count; i++ {
		xa, ya, xb, yb := randClassAB(*radius, *width, *distance)
		if err := w.Write(toStrings([]float64{xa, ya, xb, yb})); err != nil {
			log.Fatalln("Error writing record to csv:", err)
		}
	}
	w.Flush()
	if err := w.Error(); err != nil {
		log.Fatal(err)
	}
}
