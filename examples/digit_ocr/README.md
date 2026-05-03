# Digit OCR Example

## How to train?

### Generate data

- sklearn dataset

  ```
  cd data/sklearn
  python gendata.py
  ```

- mnist dataset

  ```
  cd data/mnist
  python gendata.py
  ```

  The mnist data file `mnist.pkg.gz` is copied from
  [this repo](https://github.com/mnielsen/neural-networks-and-deep-learning/blob/master/data/mnist.pkl.gz)

### Train model

- Use computation graph approach:

  ```
  go run .
  ```

- Use precomputed gradient formula (array-based MLP) approach:

  ```
  go run . --arr
  ```

## Model Illustration

![Digit OCR Model](/assets/digit-ocr-model.png)
