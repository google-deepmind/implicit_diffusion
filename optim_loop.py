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

"""Bilevel optimization loop through sampling with Langevin dynamics."""

import functools
from typing import Any, Callable

import jax
import jax.numpy as jnp
import numpy as np

import updates


def optim_loop(
    rng_seed: int,
    gamma: float,
    lamb_kl: float,
    bs: int,
    bs_ref: int,
    dim_x: int,
    steps: int,
    eta: float,
    log_option: bool,
    print_option: bool,
    print_every: int,
    theta_init: Any,
    energy_func: Callable[[Any, Any], Any],
    reward_func: Callable[[Any], Any],
    eta_func: Callable[[Any, Any], Any],
) -> Any:
  """Optimization loop for the Implicit Diffusion algorithm.

  Reproduces the results on Langevine dynamics for Implicit Diffusion, section
  5.1. in (Marion et al. 2024).

  Args:
    rng_seed: The seed for the random number generator.
    gamma: The learning rate for the first-order Langevin dynamics.
    lamb_kl: Strength of the KL divergence term.
    bs: The batch size for the batch in the first-order Langevin dynamics.
    bs_ref: The batch size for the reference distribution.
    dim_x: The dimension of the state space.
    steps: The number of optimization steps.
    eta: Constant for the learning rate for the parameter theta.
    log_option: Whether to log the reward and KL divergence at each step.
    print_option: Whether to print the reward and KL divergence at some steps.
    print_every: The frequency at which to print reward and KL gradient norm.
    theta_init: The initial value of the theta parameter.
    energy_func: The energy function for the Langevin dynamics.
    reward_func: The reward function for the optimization over theta.
    eta_func: The function that computes the learning rate over steps.

  Returns:
    hist_reward: A Numpy array of size (steps,) containing the values of the
      reward function.
    hist_kl_grad: A Numpy array of size (steps,) containing the values of the
      KL gradient norm.
    hist_theta: A Numpy array of size (steps, 6), the dynamics of theta.

  References:
    Marion et al., `Implicit Diffusion: Efficient Optimization through
    Stochastic Sampling
    <https://arxiv.org/abs/2402.05468>`_, 2024
  """
  grad_v_x = jax.vmap(jax.grad(energy_func), in_axes=(0, None))
  grad_v_theta = jax.vmap(jax.grad(energy_func, argnums=1), in_axes=(0, None))
  batch_reward = jax.jit(jax.vmap(reward_func))
  update_x = jax.jit(functools.partial(updates.update_x,
                                       gamma=gamma,
                                       grad_v_x=grad_v_x))
  update_ref = jax.jit(updates.update_ref)
  update_theta = jax.jit(
      functools.partial(updates.update_theta,
                        lamb_kl=lamb_kl,
                        bs=bs,
                        grad_v_theta=grad_v_theta,
                        batch_reward=batch_reward)
  )
  eta_func = jax.jit(eta_func)
  rng = jax.random.PRNGKey(rng_seed)
  rngs = jax.random.split(rng, 3)
  x = jax.random.normal(rngs[1], (bs, dim_x))
  x_ref = jax.random.normal(rngs[2], (bs_ref, dim_x))
  theta = theta_init
  hist_reward = []
  hist_kl_grad = []
  hist_theta = []

  for t in range(steps):
    rngs = jax.random.split(rngs[0], num=3)
    x = update_x(rngs[1], x, theta)
    x_ref = update_ref(rngs[2], x_ref)
    eta_t = eta_func(eta, t)
    theta, update_kl = update_theta(x, theta, x_ref, eta_t)
    if log_option:
      hist_reward.append(jnp.mean(batch_reward(x)))
      hist_kl_grad.append(jnp.linalg.norm(update_kl))
      hist_theta.append(theta)

    if print_option:
      if t % print_every == 0:
        print(
            f'step {t} / {steps} +++ reward = {hist_reward[-1]:.3f} +++ KL ='
            f' {hist_kl_grad[-1]:.3f}'
        )
  if log_option:
    hist_reward = np.array(hist_reward)
    hist_kl_grad = np.array(hist_kl_grad)
    hist_theta = np.array(hist_theta)
  return hist_reward, hist_kl_grad, hist_theta
