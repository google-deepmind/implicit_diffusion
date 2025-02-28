# Implicit Diffusion

This is an implementation of the Implicit Diffusion algorithm, reproducing some
experimental results from the paper ([Implicit Diffusion: Efficient Optimization
through Stochastic Sampling](https://arxiv.org/abs/2402.05468), Marion et al.,
2024).

The code is written in JAX, and reproduces the results of the paper on reward
training of Langevin processes (section 5.1)

## Usage

In order to run the code, you can copy these files to your local machine and run
the following command, with at least the two following required flags:

```$ python main.py --file_path=your_file_path --expe_name=your_expe_name```

This will save the results in the folder `your_file_path/your_expe_name` (which
should already exist), in the form of three .npy files.

These files contain the following information:

- `hist_reward.npy`, a Numpy array of shape `(steps,)` containing the value of
the reward at each step.
- `hist_kl_grad.npy`, a Numpy array of shape `(steps,)` containing the value of
the norm of the gradient for the KL divergence at each step.
- `hist_theta.npy`, a Numpy array of shape `(steps, 6)` containing the value of
the parameters of the Langevin process at each step.

## Citing this work

In order to cite this work in your own work, you can use the following citation:

```
@article{implicitdiffusion,
      title={Implicit Diffusion: Efficient Optimization through Stochastic Sampling},
      author={Marion, Pierre and Korba, Anna and Bartlett, Peter and Blondel, Mathieu and De Bortoli, Valentin and Doucet, Arnaud and Llinares-L{\'o}pez, Felipe and Paquette, Courtney and Berthet, Quentin},
      year={2024},
}
```

## License and disclaimer

Copyright 2025 Google LLC

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.

