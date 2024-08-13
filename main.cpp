#include <iostream>
#include "constants.hpp"
#include "NeuralNetwork.hpp"

int main() {
  NeuralNetwork<
    NR_INPUTS,
    NR_HIDDEN_NODES,
    NR_OUTPUTS,
    NR_TRAINING_SETS
  > nn;

  nn.set_learning_rate(LEARNING_RATE);
  nn.set_training_data(BINARY_OPERATOR_INPUTS, XOR_OUTPUTS);
  nn.train();
}