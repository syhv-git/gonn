# gonn

This package currently contains a generalized dropout feed-forward neural network implementation. Any future neural network models I create will be added to this library. The code was originally generated through OpenAI's ChatGPT, which was further debugged and enriched by my own analysis.

## Installation

To use this package, run the following command in the directory containing your `go.mod` file:

```go get -ut github.com/syhv-git/gonn```

This package is not currently thread-safe tested

## Usage

Refer to the test file for proper usage. I have also added some documentation for important public methods. The epochs are managed by the user and it is up to the user on whether the dropout rate is set only once or every epoch when training.
> The Dropout rate must be set to 0 when performing validation and predictions on new data

## Issues

If there are any bugs, or further functionality is requested, please make a pull request with suitable information for reproducibility.
