#pragma once

const size_t NR_INPUTS = 2;
const size_t NR_HIDDEN_NODES = 2;
const size_t NR_OUTPUTS = 1;
const size_t NR_TRAINING_SETS = 4;

const double LEARNING_RATE = 0.1;

const double BINARY_OPERATOR_INPUTS[4][2] = {
  {0, 0},
  {1, 0},
  {0, 1},
  {1, 1}
};

const double XOR_OUTPUTS[4][1] = {
  {0},
  {1},
  {1},
  {0}
};

const double OR_OUTPUTS[4][1] = {
  {0},
  {1},
  {1},
  {1}
};

const double AND_OUTPUTS[4][1] = {
  {0},
  {0},
  {0},
  {1}
};
