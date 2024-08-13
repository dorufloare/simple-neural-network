#include <cstddef>
#include <cmath>
#include <cstdlib>
#include <numeric>
#include <algorithm>
#include <iostream>
#include <cstring>
#include <random> 
#include "constants.hpp"

template <
  std::size_t NR_INPUTS, 
  std::size_t NR_HIDDEN_NODES, 
  std::size_t NR_OUTPUTS,
  std::size_t NR_TRAINING_SETS
>
class NeuralNetwork {
  public:
    NeuralNetwork() 
      : learning_rate(0.01), 
        nr_epochs(NR_EPOCHS) 
    {
      std::iota(training_set_order, training_set_order + NR_TRAINING_SETS, 0);
      initialize_weights();
    }

    void set_learning_rate(double learning_rate) {
      this->learning_rate = learning_rate;
    }

    void set_training_inputs(const double (&training_inputs)[NR_TRAINING_SETS][NR_INPUTS]) {
      std::memcpy(this->training_inputs, training_inputs, NR_TRAINING_SETS * NR_INPUTS * sizeof(double));
    }

    void set_training_outputs(const double (&training_outputs)[NR_TRAINING_SETS][NR_OUTPUTS]) {
      std::memcpy(this->training_outputs, training_outputs, NR_TRAINING_SETS * NR_OUTPUTS * sizeof(double));
    }
    
    void set_training_data(
      const double (&training_inputs)[NR_TRAINING_SETS][NR_INPUTS],
      const double (&training_outputs)[NR_TRAINING_SETS][NR_OUTPUTS]) 
    {     
      set_training_inputs(training_inputs);
      set_training_outputs(training_outputs);
    }

    void initialize_weights() {
      for (std::size_t i = 0; i < NR_INPUTS; ++i) 
        for (std::size_t j = 0; j < NR_HIDDEN_NODES; ++j) 
          hidden_weights[i][j] = get_random_weight();

      for (std::size_t i = 0; i < NR_HIDDEN_NODES; ++i) 
        for (std::size_t j = 0; j < NR_OUTPUTS; ++j) 
          output_weights[i][j] = get_random_weight();

      for (std::size_t i = 0; i < NR_HIDDEN_NODES; ++i) 
        hidden_layer_bias[i] = get_random_weight();

      for (std::size_t i = 0; i < NR_OUTPUTS; ++i) 
        output_layer_bias[i] = get_random_weight();
    }

    void print_network_data(std::size_t i) const {
      std::cout << "Input: ";
      for (std::size_t j = 0; j < NR_INPUTS; ++j)
        std::cout << training_inputs[i][j] << " ";
      std::cout << '\n';

      std::cout << "Output: ";
      for (std::size_t j = 0; j < NR_OUTPUTS; ++j)
        std::cout << output_layer[j] << " ";
      std::cout << '\n';

      std::cout << "Expected: " ;
      for (std::size_t j = 0; j < NR_OUTPUTS; ++j)
        std::cout << training_outputs[i][j] << " ";
      std::cout << '\n';
    }

    void process_epoch() {
      shuffle_training_set();

      for (const int i : training_set_order) {
        pass_forward(i);
        print_network_data(i);
        propagate_backwards(i);
      }
    }

    void pass_forward(const int i) {
      for (std::size_t j = 0; j < NR_HIDDEN_NODES; ++j) {
        double activation = hidden_layer_bias[j];

        for (std::size_t k = 0; k < NR_INPUTS; ++k) 
          activation += training_inputs[i][k] * hidden_weights[k][j];  
        hidden_layer[j] = sigmoid(activation);
      }

      for (std::size_t j = 0; j < NR_OUTPUTS; ++j) {
        double activation = output_layer_bias[j];

        for (std::size_t k = 0; k < NR_HIDDEN_NODES; ++k) 
          activation += hidden_layer[k] * output_weights[k][j];
        output_layer[j] = sigmoid(activation);
      }  
    }

    void propagate_backwards(const int i) {
      double delta_output[NR_OUTPUTS];

      for (std::size_t j = 0; j < NR_OUTPUTS; ++j) {
        double error = training_outputs[i][j] - output_layer[j];
        delta_output[j] = error * sigmoid_derivative(output_layer[j]);
      }

      double delta_hidden[NR_HIDDEN_NODES];

      for (std::size_t j = 0; j < NR_HIDDEN_NODES; ++j) {
        double error = 0.0;
        for (std::size_t k = 0; k < NR_OUTPUTS; ++k)
          error += delta_output[k] * output_weights[j][k];

        delta_hidden[j] = error * sigmoid_derivative(hidden_layer[j]);
      }

      for (std::size_t j = 0; j < NR_OUTPUTS; ++j) {
        output_layer_bias[j] += delta_output[j] * learning_rate;

        for (std::size_t k = 0; k < NR_HIDDEN_NODES; ++k) 
          output_weights[k][j] += hidden_layer[k] * delta_output[j] * learning_rate;
        
      }

      for (std::size_t j = 0; j < NR_HIDDEN_NODES; ++j) {
        hidden_layer_bias[j] += delta_hidden[j] * learning_rate;

        for (std::size_t k = 0; k < NR_INPUTS; ++k)
          hidden_weights[k][j] += training_inputs[i][k] * delta_hidden[j] * learning_rate;
      }
    }

    void train() {
      for (int epoch = 0; epoch < nr_epochs; ++epoch) 
        process_epoch();
    }

  private:
    double learning_rate;
    double hidden_layer[NR_HIDDEN_NODES];
    double output_layer[NR_OUTPUTS];
    double hidden_layer_bias[NR_HIDDEN_NODES]; 
    double output_layer_bias[NR_OUTPUTS];
    double hidden_weights[NR_INPUTS][NR_HIDDEN_NODES];
    double output_weights[NR_HIDDEN_NODES][NR_OUTPUTS];
    double training_inputs[NR_TRAINING_SETS][NR_INPUTS];
    double training_outputs[NR_TRAINING_SETS][NR_OUTPUTS];

    int training_set_order[NR_TRAINING_SETS];
    int nr_epochs;

    double get_random_weight() {
      return static_cast<double>(rand()) / RAND_MAX;
    }

    double sigmoid(double x) const {
      return 1.0 / (1.0 + exp(-x));
    }

    double sigmoid_derivative(double x) const {
      return x * (1.0 - x);
    }

    void shuffle_training_set() {
      std::random_device rd;
      std::mt19937 g(rd());
      std::shuffle(training_set_order, training_set_order + NR_TRAINING_SETS, g);
    }
};
