# NuPerturb: Neutrino Mass Matrix Perturbation Analysis

**NuPerturb** is a computational framework for exploring the effects of small perturbations in the neutrino mass matrix on observable parameters, including the PMNS mixing angles, the Dirac CP phase (δ_CP), and the Jarlskog invariant (J). This project integrates analytical formalism with Monte Carlo simulations to study the stability and sensitivity of neutrino mixing under microscopic deviations.

## Features

- Monte Carlo sampling of perturbed Majorana mass matrices.
- Automatic extraction of PMNS mixing angles (θ12, θ13, θ23) and δ_CP.
- Calculation of mass-squared differences (Δm²_21, Δm²_31) for normal or inverted ordering.
- Computation of the Jarlskog invariant as a measure of CP violation.
- Statistical analysis including mean, variance, correlation, and stability indices.
- Visualization-ready output for histograms and correlation plots.

## Requirements

- Python 3.8+
- NumPy
- SciPy
- Matplotlib
- Pandas (optional, for advanced data analysis)

## Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/NuPerturb.git
cd NuPerturb
Install dependencies:

bash
Copy code
pip install numpy scipy matplotlib pandas
Usage
Define the base neutrino mass matrix M_base.

Set the perturbation amplitude sigma and number of Monte Carlo samples N_samples.

Run the main simulation script to generate perturbed matrices, extract mixing parameters, and compute observables.

Visualize results using Python plotting routines.

Example:

python
Copy code
from nuperturb import delta_M, extract_angles

# Monte Carlo loop
for k in range(N_samples):
    M_prime = M_base + delta_M()
    eigvals, eigvecs = np.linalg.eigh(M_prime)
    theta_12, theta_13, theta_23, delta = extract_angles(eigvecs)
License
This project is licensed under the MIT License.
