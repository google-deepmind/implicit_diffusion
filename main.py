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

"""Reproducing the results on Langevine dynamics for Implicit Diffusion."""

import time

from absl import app
from absl import flags
import jax.numpy as jnp
import numpy as np


import energy
import optim_loop
import reward
import updates


_RNG_SEED = flags.DEFINE_integer('rng_seed', 0, 'Random number generator seed.')
_BS = flags.DEFINE_integer('bs', 1_000, 'Batch size.')
_BS_REF = flags.DEFINE_integer('bs_ref', 1, 'Batch size for ref.')
_STEPS = flags.DEFINE_integer('steps', 5_000, 'Number of steps.')
_GAMMA = flags.DEFINE_float('gamma', 0.2, 'Step size constant for sampling')
_ETA = flags.DEFINE_float('eta', 0.1, 'Step size constant for training.')
_DIM_X = flags.DEFINE_integer('dim_x', 2, 'dimension of x')
_LAMB_KL = flags.DEFINE_float('lamb_kl', 0., 'KL weight.')

_LOG_OPTION = flags.DEFINE_bool('log_option', True, 'Log option.')
_PRINT_OPTION = flags.DEFINE_bool('print_option', True, 'Print option.')
_SAVE_OPTION = flags.DEFINE_bool('save_option', True, 'Save checkpoint.')
_PRINT_EVERY = flags.DEFINE_integer('print_every', 500, 'Print every N steps.')
_EXPE_NAME = flags.DEFINE_string('expe_name',
                                 None,
                                 'Experiment name.',
                                 required=True)
_FILE_PATH = flags.DEFINE_string(
    'file_path',
    None,
    'File path for saving results.',
    required=True)


def main(unused_argv) -> None:

  rng_seed = _RNG_SEED.value
  bs = _BS.value
  bs_ref = _BS_REF.value
  steps = _STEPS.value
  gamma = _GAMMA.value
  eta = _ETA.value
  lamb_kl = _LAMB_KL.value
  dim_x = _DIM_X.value

  energy_func = energy.hex_mixture
  reward_func = reward.reward_hard
  eta_func = updates.eta_func_const

  theta_init = jnp.array([1.5, 0., 1.5, 0., 1.5, 0.])

  log_option = _LOG_OPTION.value
  print_option = _PRINT_OPTION.value
  print_every = _PRINT_EVERY.value

  tic = time.time()
  hist_reward, hist_kl_grad, hist_theta = optim_loop.optim_loop(
      rng_seed,
      gamma,
      lamb_kl,
      bs,
      bs_ref,
      dim_x,
      steps,
      eta,
      log_option,
      print_option,
      print_every,
      theta_init,
      energy_func=energy_func,
      reward_func=reward_func,
      eta_func=eta_func,
  )
  toc = time.time()
  print(f'Total time, {steps} steps: {toc - tic:.3f}s')

  if _SAVE_OPTION.value:
    chkpt_dir = _FILE_PATH.value + _EXPE_NAME.value
    np.save(chkpt_dir + '/hist_theta.npy', hist_theta)
    np.save(chkpt_dir + '/hist_reward.npy', hist_reward)
    np.save(chkpt_dir + '/hist_kl_grad.npy', hist_kl_grad)


if __name__ == '__main__':
  app.run(main)
