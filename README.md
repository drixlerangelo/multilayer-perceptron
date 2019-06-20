# Multilayer Perceptron

---

## Description
This is an implementation of a Multilayer Perceptron from scratch using the NumPy library.

## Usage
To get started, you can use the application by running this command below to the terminal
```
python main.py
```
### Options
There are a few options that you may want to use. For example, one can set the epochs by:
```
python main.py --epochs=1000
```
Here are the options:
* `training_data` A name of a JSON file where you store your training set. Default: `training_data.json`
* `testing_data` A name of a JSON file where you store your testing set. Default: `testing_data.json`
* `architecture_file` A name of a JSON file that stores the architecture of the Neural Network. Default: `neural_network_architecture.json`
* `output_file` A generated JSON file where the predictions from running the application is stored. Default: `predictions.json`
* `epochs` A number where how many times training repeats to optimize the parameters. Default: `100`
