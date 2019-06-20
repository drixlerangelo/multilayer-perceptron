import argparse
from model import NeuralNetwork
from tools import JsonProcessor


def get_console_args():
    """Converts console arguments into namespace

    :rtype: argparse.Namespace
    """

    parser = argparse.ArgumentParser()

    parser.add_argument('--training_data', type=str, default='training_data.json')
    parser.add_argument('--testing_data', type=str, default='testing_data.json')
    parser.add_argument('--architecture_file', type=str, default='neural_network_architecture.json')
    parser.add_argument('--output_file', type=str, default='predictions.json')
    parser.add_argument('--epochs', type=int, default=100)

    return parser.parse_args()


def main():
    """Main entry to the application
    """

    args = get_console_args()

    # Define the network's architecture
    architecture = JsonProcessor.load(args.architecture_file)

    # Initialize the Neural Network
    nn = NeuralNetwork(**architecture)

    # Set its parameters
    nn.set_parameters()

    # Loads the data
    nn.load_data(args.training_data)

    # Train the neural network
    nn.train(args.epochs)

    # Get its predictions
    predictions = nn.predict(JsonProcessor.load(args.testing_data)['input'])

    # Save its predictions
    JsonProcessor.beautify(args.output_file, {
        'output': predictions.tolist()
    })


if __name__ == '__main__':
    main()
