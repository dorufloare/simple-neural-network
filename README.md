## SIMPLE NEURAL NETWORK 

This project implements a simple feedforward neural network with backpropagation. The network is highly configurable, allowing you to specify the number of input nodes, hidden nodes, output nodes, and training sets. The neural network is trained using gradient descent with a configurable learning rate.

# HOW TO USE

## 1. Include the necessary headers

```
#include <iostream> 
#include "constants.hpp"
#include "NeuralNetwork.hpp"
```

## 2. Initialize the NeuralNetwork class

```
NeuralNetwork<
  NR_INPUTS,
  NR_HIDDEN_NODES,
  NR_OUTPUTS,
  NR_TRAINING_SETS
> nn;
```

*If you are using this neural network for bitwise operations (XOR, AND, OR), these constants are recommended

```
const size_t NR_INPUTS = 2;
const size_t NR_HIDDEN_NODES = 2;
const size_t NR_OUTPUTS = 1;
const size_t NR_TRAINING_SETS = 4;
```

## 3. Set training data

```
nn.set_training_data(TRAINING_INPUT, TRAINING_OUTPUT)
```

*where TRAINING_INPUT is a 2D-array with sizes [NR_TRAINING_SETS][NR_INPUTS] and TRAINING_OUPUT is another 2D array with sizes [NR_TRAINING_SETS][NR_OUTPUTS]

*if you want to train the network to make bit operations, I have provided you with the training data in the constants.hpp file

## 4. Train the neural network

```
nn.train(NR_OF_EPOCHS)
```

*I recommend to train it with at least 10000 epochs for accurate results, but you can play around with smaller values to visualise the evolution, at the end of the training, the neural network data will be printed, showing the inputs, outputs and the expected outputs

## 5. Compile and run
   
