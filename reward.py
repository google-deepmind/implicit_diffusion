# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Reward functions for optimization through sampling with Langevin dynamics."""

import jax
import jax.numpy as jnp


def reward_hard(x: jnp.ndarray):
  """An example of a reward function over indivual points.

  Used in reward optimzation for Langevin dynamics in Implicit Diffusion,
  section 5.1. in (Marion et al. 2024).

  Args:
    x: A point in the state space.
  Returns:
    The reward at the point.
  """
  center_reward = jnp.array([4, 3.8])
  tau = 20
  alpha = 2.5
  return (
      (x[0] >= 0).astype(float)
      * jnp.exp(
          -((x[0] - center_reward[0]) ** 2) / tau
          - (x[1] - center_reward[1]) ** 2 / tau
      )
      / alpha
  )


def reward_smooth(x: jnp.ndarray):
  """An example of a reward function over indivual points - smoothed version.

  Used in reward optimzation for Langevin dynamics in Implicit Diffusion,
  section 5.1. in (Marion et al. 2024).

  Args:
    x: A point in the state space.
  Returns:
    The reward at the point.
  """
  center_reward = jnp.array([4, 3.8])
  tau = 20
  lamb_reg = 0.5
  alpha = 0.7
  return (
      jax.nn.sigmoid(x[0] / lamb_reg)
      * jnp.exp(
          -((x[0] - center_reward[0]) ** 2) / tau
          - (x[1] - center_reward[1]) ** 2 / tau
      )
      / alpha
  )
