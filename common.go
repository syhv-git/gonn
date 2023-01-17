package gonn

import (
	"encoding/csv"
	"log"
	"os"
	"strconv"
)

// LoadCSVData loads training data from source and returns two float64 matrices containing the training data and the target outcome.
// Ensure the CSV records do not contain commas (i.e. 10,000 should be 10000) nor quotation mark delimiters
//
// inputLen determines how many fields contain the input data.
//
// outputLen determines the number of targets.
//
// names signifies if the first entry in the CSV contains field names.
func LoadCSVData(src string, inputLen, targetLen int, names bool) ([][]float64, [][]float64) {
	entryLen := inputLen + targetLen
	f, err := os.Open(src)
	if err != nil {
		log.Fatal(err.Error())
	}
	defer f.Close()

	rdr := csv.NewReader(f)
	rdr.FieldsPerRecord = entryLen
	data, err := rdr.ReadAll()
	if err != nil {
		log.Fatal(err.Error())
	}

	if names {
		data = data[1:]
	}
	inputs, targets := make([][]float64, len(data)), make([][]float64, len(data))

	for k, x := range data {
		inputs[k], targets[k] = make([]float64, inputLen), make([]float64, targetLen)
		i, j := 0, 0

		for n, y := range x {
			d, err := strconv.ParseFloat(y, 64)
			if err != nil {
				log.Fatal(err.Error())
			}

			if n < inputLen {
				inputs[k][i] = d
				i++
				continue
			}
			targets[k][j] = d
			j++
		}
	}

	return inputs, targets
}
