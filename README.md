# LG-Beam Induced Skyrmions: sLLG Simulation on a Frustrated Triangular Lattice

This repository includes a simulation script (`simulate.py`) and a requirements file (`requirements.txt`) to run the script. This script was used to generate the results of the paper entitled “Skyrmion generation via Laguerre-Gaussian beam irradiation in frustrated magnets,” which is currently under submission to Physical Review B. A pre-print of this paper is available at arXiv: https://arxiv.org/abs/2603.03773. The script is made available to the public in accordance to the Data Availability Statement of Physical Review B.

## Requirements

To run `simulate.py`, the following packages are required: numpy, pandas, scipy, and tqdm. All of these packages can be installed using pip:

```bash
pip install -r requirements.txt
```

## Simulation

The simulation script starts with a ferromagnetic spin configuration in a triangular lattice. A Laguerre-Gaussian (LG) beam is applied to the lattice as a thermal profile and spin dynamics are observed through the stochastic Landau–Lifshitz–Gilbert (sLLG) equation. After the simulation, the final spin configuration, scalar spin chirality, and spin structure factor are plotted.