import numpy as np

class NeuralNetwork:
    def __init__(self, layers, learning_rate=0.1, epochs=1000, initial_weights=None):
        self.layers = layers
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.weights = []
        self.biases = []

        # Initialize weights and biases for each layer
        for i in range(len(layers) - 1):
            if initial_weights and len(initial_weights) > i:
                # Use user-provided weights if available
                self.weights.append(np.array(initial_weights[i]))
            else:
                raise ValueError(f"Initial weights for layer {i} are required.")

            # Initialize biases randomly
            self.biases.append(np.random.randn(layers[i + 1]))

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def sigmoid_derivative(self, x):
        return x * (1 - x)

    def forward_propagation(self, inputs):
        activations = [inputs]
        for i in range(len(self.weights)):
            net_input = np.dot(activations[i], self.weights[i]) + self.biases[i]
            activation = self.sigmoid(net_input)
            activations.append(activation)
        return activations

    def backward_propagation(self, activations, expected_output):
        error = expected_output - activations[-1]
        deltas = [error * self.sigmoid_derivative(activations[-1])]

        for i in reversed(range(len(self.weights) - 1)):
            delta = np.dot(deltas[0], self.weights[i + 1].T) * self.sigmoid_derivative(activations[i + 1])
            deltas.insert(0, delta)

        for i in range(len(self.weights)):
            self.weights[i] += self.learning_rate * np.dot(activations[i].T, deltas[i])
            self.biases[i] += self.learning_rate * np.sum(deltas[i], axis=0)

    def train(self, training_inputs, training_outputs):
        for epoch in range(self.epochs):
            activations = self.forward_propagation(training_inputs)
            self.backward_propagation(activations, training_outputs)

    def predict(self, inputs):
        activations = self.forward_propagation(inputs)
        output = activations[-1]
        return np.round(output)


def get_user_inputs():
    layers = list(map(int, input("Enter the number of neurons in each layer (space-separated): ").split()))
    learning_rate = float(input("Enter the learning rate: "))
    epochs = int(input("Enter the number of epochs: "))

    print("\nPlease provide the initial weights for each layer.")
    initial_weights = []
    for i in range(len(layers) - 1):
        print(f"For layer {i + 1}, enter weights (as a {layers[i]}x{layers[i + 1]} matrix):")
        weights_input = []
        for _ in range(layers[i]):
            weights_input.append(list(map(float, input(f"Row {_ + 1}: ").split())))
        initial_weights.append(weights_input)

    print("\nEnter training data (inputs and outputs). Example for AND gate:\nInputs: 0 0, 0 1, 1 0, 1 1\nOutputs: 0, 0, 0, 1")
    training_inputs = []
    print("Enter input rows one by one (e.g., '0 0' for each input pair):")
    for _ in range(4):  # Example assumes 4 training samples
        training_inputs.append(list(map(int, input().split())))
    training_inputs = np.array(training_inputs)

    training_outputs = list(map(int, input("Enter the corresponding output values (space-separated): ").split()))
    training_outputs = np.array(training_outputs).reshape(-1, 1)

    return layers, learning_rate, epochs, initial_weights, training_inputs, training_outputs


def main():
    layers, learning_rate, epochs, initial_weights, training_inputs, training_outputs = get_user_inputs()

    nn = NeuralNetwork(layers, learning_rate, epochs, initial_weights)

    print("\nTraining the neural network...")
    nn.train(training_inputs, training_outputs)
    print("\nTraining complete.")

    while True:
        user_input = input("\nEnter test inputs (space-separated, e.g., '1 0') or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        test_data = np.array(list(map(int, user_input.split()))).reshape(1, -1)
        output = nn.predict(test_data)
        print("Predicted output:", int(output[0][0]))


if __name__ == "__main__":
    main()
