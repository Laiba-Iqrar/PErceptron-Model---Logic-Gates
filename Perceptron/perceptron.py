import numpy as np

class Perceptron:
    def __init__(self, input_size, learning_rate=0.1, epochs=100, weights=None, bias=0):
        # Initialize weights and bias
        self.learning_rate = learning_rate
        self.epochs = epochs
        if weights is None:
            self.weights = np.zeros(input_size)
        else:
            if len(weights) != input_size:
                raise ValueError(f"The number of weights should be {input_size}.")
            self.weights = np.array(weights)
        self.bias = bias

    def activation_function(self, x):
        # Step activation function
        return 1 if x >= 0 else 0

    def predict(self, inputs):
        # Predict output for a given input
        total_activation = np.dot(inputs, self.weights) + self.bias
        return self.activation_function(total_activation)

    def train(self, training_inputs, labels):
        # Train the perceptron
        for epoch in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction = self.predict(inputs)
                # Update weights and bias if there is an error
                error = label - prediction
                self.weights += self.learning_rate * error * inputs
                self.bias += self.learning_rate * error

def get_user_inputs():
    # Get user inputs for perceptron setup
    input_size = int(input("Enter the number of inputs for the perceptron: "))
    learning_rate = float(input("Enter the learning rate (e.g., 0.1): "))
    epochs = int(input("Enter the number of epochs: "))

    # Get weights and bias
    print("\nSet the initial weights:")
    weights = list(map(float, input(f"Enter {input_size} weights (space-separated): ").split()))
    bias = float(input("Enter the initial bias: "))

    # Get logical gate inputs and expected outputs
    print("\nEnter the truth table for the logical gate (one row per line):")
    print("Example for an AND gate with two inputs:")
    print("0 0 0")
    print("0 1 0")
    print("1 0 0")
    print("1 1 1")

    training_data = []
    for _ in range(2**input_size):
        row = list(map(int, input().split()))
        training_data.append(row)

    training_inputs = np.array([data[:input_size] for data in training_data])
    labels = np.array([data[input_size] for data in training_data])

    return input_size, learning_rate, epochs, weights, bias, training_inputs, labels

def main():
    # Get parameters from the user
    input_size, learning_rate, epochs, weights, bias, training_inputs, labels = get_user_inputs()

    # Create the perceptron
    perceptron = Perceptron(input_size, learning_rate=learning_rate, epochs=epochs, weights=weights, bias=bias)

    # Train the perceptron
    perceptron.train(training_inputs, labels)
    print("\nTraining complete.")
    print("Final weights:", perceptron.weights)
    print("Final bias:", perceptron.bias)

    # Test the perceptron
    while True:
        user_input = input("\nEnter binary inputs to test (space-separated) or 'exit' to quit: ")
        if user_input.lower() == 'exit':
            break
        test_data = list(map(int, user_input.split()))
        if len(test_data) != input_size:
            print(f"Please enter exactly {input_size} inputs.")
            continue
        output = perceptron.predict(np.array(test_data))
        print("Predicted output:", output)

if __name__ == "__main__":
    main()
