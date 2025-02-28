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

"""Energy functions for Langevin dynamics."""

import jax
import jax.numpy as jnp


def hex_mixture(x: jnp.ndarray, theta: jnp.ndarray, sig_sq: float = 2.0):
  """Energy function for an hexagonal Gaussian mixture."""
  v = jax.nn.softmax(theta)
  center1 = 4 * jnp.array([1., 0.])
  center2 = 4 * jnp.array([1/2, jnp.sqrt(3)/2])
  center3 = 4 * jnp.array([-1/2, jnp.sqrt(3)/2])
  center4 = -center1
  center5 = -center2
  center6 = -center3
  return -jnp.log(
      v[0] * jnp.exp(jnp.sum(-((x - center1) ** 2)) / sig_sq)
      + v[1] * jnp.exp(jnp.sum(-((x - center2) ** 2)) / sig_sq)
      + v[2] * jnp.exp(jnp.sum(-((x - center3) ** 2)) / sig_sq)
      + v[3] * jnp.exp(jnp.sum(-((x - center4) ** 2)) / sig_sq)
      + v[4] * jnp.exp(jnp.sum(-((x - center5) ** 2)) / sig_sq)
      + v[5] * jnp.exp(jnp.sum(-((x - center6) ** 2)) / sig_sq)
  )


def hex_random(rng: jnp.ndarray, x_ref: jnp.ndarray, sig_sq: float = 2.0):
  """Random sample from a hexagonal Gaussian mixture."""
  center1 = 4 * jnp.array([1., 0.])
  center2 = 4 * jnp.array([1/2, jnp.sqrt(3)/2])
  center3 = 4 * jnp.array([-1/2, jnp.sqrt(3)/2])
  center4 = -center1
  center5 = -center2
  center6 = -center3
  means_x = jnp.array([center1, center2, center3, center4, center5, center6])
  theta_init = jnp.array([1.5, 0., 1.5, 0., 1.5, 0.])
  probas = jax.nn.softmax(theta_init)
  bs = x_ref.shape[0]
  rngs = jax.random.split(rng, 2)
  idxs = jax.random.choice(rngs[0], 6, shape=(bs,), p=probas)
  return means_x[idxs] + sig_sq * jax.random.normal(
      rngs[1], shape=means_x[idxs].shape
  )
