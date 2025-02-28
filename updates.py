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

"""Updates for bilevel Langevin dynamics and parameter optimization."""

import jax
import jax.numpy as jnp

import energy


def update_x(rng, x, theta, gamma, grad_v_x):
  z = jax.random.normal(rng, x.shape)
  x = x - gamma * grad_v_x(x, theta) + jnp.sqrt(2 * gamma) * z
  return x


def update_ref(rng, x_ref):
  z = energy.hex_random(rng, x_ref)
  return z


def update_theta(x, theta, x_ref, eta, lamb_kl, bs, grad_v_theta, batch_reward):
  mean_x = jnp.mean(grad_v_theta(x, theta), axis=0)
  mean_ref = jnp.mean(grad_v_theta(x_ref, theta), axis=0)
  update_reward = -batch_reward(x) @ grad_v_theta(
      x, theta
  ) / bs + jnp.mean(batch_reward(x)) * mean_x
  update_kl = mean_ref - mean_x
  theta = theta + eta * (update_reward + lamb_kl * update_kl)
  return theta, update_kl


def eta_func_sqrt(eta, t):
  return eta / jnp.sqrt(t + 1)


def eta_func_const(eta, t):
  return eta + 0 * t
