# Copyright 2019 DeepMind Technologies Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Simple network classes for Tensorflow based on tf.Module."""

import math
from torch import nn
# Temporarily disable TF2 behavior until code is updated.

# This code is based directly on the TF docs:
# https://www.tensorflow.org/versions/r1.15/api_docs/python/tf/Module

class MLP(nn.Module):
  """A simple dense network built from linear layers."""

  def __init__(self,
               input_size,
               hidden_sizes,
               output_size,
               activate_final=False):
    """Create the MLP.

    Args:
      input_size: (int) number of inputs
      hidden_sizes: (list) sizes (number of units) of each hidden layer
      output_size: (int) number of outputs
      activate_final: (bool) should final layer should include a ReLU
      name: (string): the name to give to this network
    """

    super(MLP, self).__init__()
    _layers = []
    for size in hidden_sizes:
      _layers.append(nn.Linear(in_features=input_size, out_features=size))
      _layers.append(nn.ReLU())
      input_size = size
    # Output layer
    _layers.append(nn.Linear(in_features=input_size, out_features=size))
    if activate_final:
      _layers.append(nn.ReLU())
    
    self.layers = nn.ModuleList(_layers)


  def forward(self, x):
    for layer in self._layers:
      x = layer(x)
    return x


class MLPTorso(nn.Module):
  """A specialized half-MLP module when constructing multiple heads.

  Note that every layer includes a ReLU non-linearity activation.
  """

  def __init__(self, input_size, hidden_sizes):
    super(MLPTorso, self).__init__()
    _layers = []
    for size in hidden_sizes:
      _layers.append(nn.Linear(in_features=input_size, out_features=size))
      _layers.append(nn.ReLU())
      input_size = size
    self.layers = nn.ModuleList(_layers)

  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    return x
