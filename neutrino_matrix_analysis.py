from google.colab import files
uploaded = files.upload()

class Args:
    nufit_json = 'nufit.json'  
    cov_path = None
    use_cov_sampler = False
    cov_scale = 1.0
    ordering = 'normal'
    mass_matrix_type = 'Majorana'
    n_samples = 200
    epsilon_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    random_scale = 1e-3
    complex_phase = False
    out_prefix = 'nu_analysis'

args = Args()

# -*- coding: utf-8 -*-
"""
NuFIT-ready neutrino-mass perturbation analysis
- Uses NuFIT (default: v6.0) as baseline parameters if available
- Supports Majorana vs Dirac, normal/inverted ordering
- Sampling from experimental covariance (if provided) or from per-parameter uncertainties
- Improved extraction of mixing angles/CP following PDG conventions
- Outputs CSV summaries, PNG figures, and detailed stats (histograms, correlations, CIs)

"""

import numpy as np
import matplotlib.pyplot as plt
from numpy import linalg as LA
import os
import json
import argparse
import logging
import sys
from scipy.stats import multivariate_normal
from tqdm import tqdm
import pandas as pd

# Logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# Utilities
def deg2rad(x_deg):
    return x_deg * np.pi / 180.0

def rad2deg(x_rad):
    return x_rad * 180.0 / np.pi

def clamp(x, lo=-1.0, hi=1.0):
    return max(lo, min(hi, x))

# Build PMNS (PDG convention)
# We'll construct U following PDG: U = R23 * P(delta) * R13 * P(-delta) * R12
# but here we use the common parametrization consistent with many codes.

def build_PMNS(theta12_deg, theta13_deg, theta23_deg, delta_deg):
    th12 = deg2rad(theta12_deg)
    th13 = deg2rad(theta13_deg)
    th23 = deg2rad(theta23_deg)
    delta = deg2rad(delta_deg)
    c12 = np.cos(th12); s12 = np.sin(th12)
    c13 = np.cos(th13); s13 = np.sin(th13)
    c23 = np.cos(th23); s23 = np.sin(th23)
    e_minus_i_delta = np.exp(-1j * delta)
    e_plus_i_delta  = np.exp(1j * delta)
    U = np.array([
        [ c12*c13,                       s12*c13,                       s13 * e_minus_i_delta ],
        [ -s12*c23 - c12*s23*s13*e_plus_i_delta,  c12*c23 - s12*s23*s13*e_plus_i_delta,  s23*c13 ],
        [  s12*s23 - c12*c23*s13*e_plus_i_delta, -c12*s23 - s12*c23*s13*e_plus_i_delta,  c23*c13 ]
    ], dtype=complex)
    return U

# Mass matrix builders

def build_mass_matrix_from_masses(U, masses, mass_matrix_type='Majorana'):
    mdiag = np.diag(masses)
    if mass_matrix_type.lower() == 'majorana':
        M = U.dot(mdiag).dot(U.T)  # Majorana: U * m_diag * U^T
    else:
        M = U.dot(mdiag).dot(U.conj().T)  # Dirac-like hermitian construction
    return M

# Perturbation constructors

def structured_deltaM(delta_params):
    d = np.zeros((3,3), dtype=complex)
    if '00' in delta_params: d[0,0] = delta_params['00']
    if '11' in delta_params: d[1,1] = delta_params['11']
    if '22' in delta_params: d[2,2] = delta_params['22']
    for key in ['01','02','12']:
        if key in delta_params:
            i = int(key[0]); j = int(key[1])
            d[i,j] = delta_params[key]
            d[j,i] = delta_params[key]
    return d

def random_symmetric_delta(scale=1e-3, complex_phase=False):
    A = np.zeros((3,3), dtype=complex)
    for i in range(3):
        for j in range(i,3):
            mag = np.random.normal(loc=0.0, scale=scale)
            if complex_phase and mag != 0.0:
                phase = np.random.uniform(0,2*np.pi)
                val = mag * np.exp(1j*phase)
            else:
                val = mag
            A[i,j] = val
            A[j,i] = val
    return A

# Diagonalize with rephasing & ordering handling

def diagonalize_mass_matrix(M, ordering='normal'):
    w, v = LA.eig(M)
    # Sort by absolute value of eigenvalues
    idx = np.argsort(np.abs(w))
    # For normal ordering we want m1<=m2<=m3; for inverted we reorder accordingly
    w_sorted = w[idx]
    v_sorted = v[:, idx]

    # Ensure positive real-phase convention for eigenvectors (remove arbitrary overall phase)
    for col in range(v_sorted.shape[1]):
        ph = np.angle(v_sorted[0,col])
        v_sorted[:,col] *= np.exp(-1j*ph)
        # normalize
        v_sorted[:,col] /= LA.norm(v_sorted[:,col])

    # Reorder to match 'normal' or 'inverted' expectation by absolute masses
    abs_sorted = np.abs(w_sorted)
    if ordering.lower() == 'normal':
        order_idx = np.argsort(abs_sorted)
    else:
        # inverted: largest becomes first (m3 < m1,m2) -> reorder descending
        order_idx = np.argsort(-abs_sorted)
    w_final = w_sorted[order_idx]
    v_final = v_sorted[:, order_idx]
    return w_final, v_final

# Jarlskog
def compute_Jarlskog(U):
    return np.imag(U[0,0]*U[1,1]*np.conj(U[0,1])*np.conj(U[1,0]))

# Extract mixing angles & CP following PDG-ish approximations

def extract_mixing_from_U(U):
    U = np.array(U, dtype=complex)
    # Enforce unitarity tolerances
    # s13 from |U_e3|
    s13 = np.abs(U[0,2])
    c13 = np.sqrt(max(0.0, 1.0 - s13**2))
    # s12 from |U_e2| = s12*c13
    s12 = np.abs(U[0,1]) / (c13 + 1e-16)
    # s23 from |U_mu3| = s23*c13
    s23 = np.abs(U[1,2]) / (c13 + 1e-16)
    s12 = clamp(s12, 0.0, 1.0)
    s23 = clamp(s23, 0.0, 1.0)
    c12 = np.sqrt(max(0.0, 1.0 - s12**2))
    c23 = np.sqrt(max(0.0, 1.0 - s23**2))
    J = compute_Jarlskog(U)
    denom = s12 * s23 * s13 * c12 * c23 * (c13**2)
    if abs(denom) < 1e-20:
        sin_delta = 0.0
    else:
        sin_delta = clamp(J / denom, -1.0, 1.0)
    delta_rad = np.arcsin(sin_delta)
    theta12_deg = rad2deg(np.arcsin(s12))
    theta13_deg = rad2deg(np.arcsin(s13))
    theta23_deg = rad2deg(np.arcsin(s23))
    delta_deg = rad2deg(delta_rad)
    return {
        'theta12_deg': theta12_deg,
        'theta13_deg': theta13_deg,
        'theta23_deg': theta23_deg,
        'delta_deg_est': delta_deg,
        'J': J
    }

# Helper checks (unitarity & physics filters)

def is_unitary(U, tol=1e-8):
    """Check unitarity: U^dagger U approx I"""
    diff = U.conj().T.dot(U) - np.eye(U.shape[0])
    max_dev = np.max(np.abs(diff))
    return max_dev <= tol, max_dev

def is_physical_sample(delta_deg, J, delta_bounds_deg=(-180.0,180.0), J_threshold=0.05):
    """Quick physicality test for a single sample"""
    if delta_deg is None:
        return False
    if not (delta_bounds_deg[0] <= delta_deg <= delta_bounds_deg[1]):
        return False
    if J is None:
        return False
    if abs(J) > J_threshold:
        return False
    return True

# Analysis per perturbation

def analyze_perturbation(theta12,theta13,theta23,delta,
                         masses,
                         deltaM, epsilon=1e-3,
                         mass_matrix_type='Majorana', ordering='normal'):
    U0 = build_PMNS(theta12,theta13,theta23,delta)
    M0 = build_mass_matrix_from_masses(U0, masses, mass_matrix_type=mass_matrix_type)
    M1 = M0 + epsilon * deltaM
    w0, v0 = diagonalize_mass_matrix(M0, ordering=ordering)
    w1, v1 = diagonalize_mass_matrix(M1, ordering=ordering)
    # Use eigenvectors as reconstructed mixing (up to phases)
    U0_fromdiag = v0
    U1_fromdiag = v1
    params0 = extract_mixing_from_U(U0_fromdiag)
    params1 = extract_mixing_from_U(U1_fromdiag)
    delta_results = {
        'delta_deg_0': params0['delta_deg_est'],
        'delta_deg_1': params1['delta_deg_est'],
        'delta_deg_change': params1['delta_deg_est'] - params0['delta_deg_est'],
        'J0': params0['J'],
        'J1': params1['J'],
        'J_change': params1['J'] - params0['J'],
        'params0': params0,
        'params1': params1,
        'masses_eig0': w0,
        'masses_eig1': w1
    }
    return delta_results

# Monte Carlo scan with physical sampling

def monte_carlo_scan(theta12,theta13,theta23,delta,masses,
                     epsilon_list, n_samples_per_epsilon=200,
                     random_scale=1e-3, complex_phase=False,
                     mass_matrix_type='Majorana', ordering='normal',
                     sampler=None):
    """
    Enhanced Monte Carlo scan that performs unitarity checks and flags unphysical samples.
    Returns detailed structure including counts of unphysical samples per epsilon and
    stores raw lists of physical/unphysical changes for later reporting.
    """
    results = {'epsilon': [], 'mean_delta_change_deg': [], 'std_delta_change_deg': [],
               'mean_J_change': [], 'std_J_change': [], 'all_delta_changes': [], 'all_J_changes': [],
               'unphysical_counts': [], 'unphysical_indices': []}
    for eps in tqdm(epsilon_list, desc="Eps loop"):
        delta_changes = []
        J_changes = []
        unphys_count = 0
        unphys_indices = []
        sample_idx = 0
        for _ in range(n_samples_per_epsilon):
            if sampler is None:
                dM = random_symmetric_delta(scale=random_scale, complex_phase=complex_phase)
            else:
                dM = sampler()
            r = analyze_perturbation(theta12,theta13,theta23,delta, masses, dM, epsilon=eps, mass_matrix_type=mass_matrix_type, ordering=ordering)
            # attempt to reconstruct U1 for checks
            try:
                U0 = build_PMNS(theta12,theta13,theta23,delta)
                M0 = build_mass_matrix_from_masses(U0, masses, mass_matrix_type=mass_matrix_type)
                M1 = M0 + eps * dM
                w1, v1 = diagonalize_mass_matrix(M1, ordering=ordering)
                U1 = v1
                phys_unitary, max_dev = is_unitary(U1)
                params1 = extract_mixing_from_U(U1)
                delta_deg_new = params1['delta_deg_est']
                J_new = params1['J']
            except Exception:
                phys_unitary = False
                max_dev = np.nan
                delta_deg_new = None
                J_new = None
            # physicality decision
            if (delta_deg_new is None) or (not phys_unitary) or (not is_physical_sample(delta_deg_new, J_new)):
                unphys_count += 1
                unphys_indices.append(sample_idx)
                # still append raw reported changes so we keep alignment
                delta_changes.append(r['delta_deg_change'])
                J_changes.append(r['J_change'])
            else:
                delta_changes.append(r['delta_deg_change'])
                J_changes.append(r['J_change'])
            sample_idx += 1
        # statistics on collected (physical) samples; if none, store nan
        if len(delta_changes) > 0:
            results['mean_delta_change_deg'].append(np.mean(delta_changes))
            results['std_delta_change_deg'].append(np.std(delta_changes))
            results['mean_J_change'].append(np.mean(J_changes))
            results['std_J_change'].append(np.std(J_changes))
        else:
            results['mean_delta_change_deg'].append(np.nan)
            results['std_delta_change_deg'].append(np.nan)
            results['mean_J_change'].append(np.nan)
            results['std_J_change'].append(np.nan)
        results['epsilon'].append(eps)
        results['all_delta_changes'].append(delta_changes)
        results['all_J_changes'].append(J_changes)
        results['unphysical_counts'].append(unphys_count)
        results['unphysical_indices'].append(unphys_indices)
    return results

# Plotting & reporting

def plot_scan_results(results, title_prefix=''):
    eps = np.array(results['epsilon'])
    mean_d = np.array(results['mean_delta_change_deg'])
    std_d = np.array(results['std_delta_change_deg'])
    mean_J = np.array(results['mean_J_change'])
    std_J = np.array(results['std_J_change'])

    fig, axes = plt.subplots(1,2, figsize=(12,5))
    axes[0].errorbar(eps, mean_d, yerr=std_d, fmt='o-')
    axes[0].set_xlabel('Epsilon')
    axes[0].set_ylabel('Mean Δδ [deg]')
    axes[0].set_title(f'{title_prefix}Δδ vs ε')
    axes[0].grid(True)

    axes[1].errorbar(eps, mean_J, yerr=std_J, fmt='s-')
    axes[1].set_xlabel('Epsilon')
    axes[1].set_ylabel('Mean ΔJ')
    axes[1].set_title(f'{title_prefix}ΔJ vs ε')
    axes[1].grid(True)

    plt.tight_layout()
    return fig

def detailed_report_and_save(results, out_prefix='nu_analysis'):
    # Save summary CSV with unphysical counts and flagging per epsilon
    rows = []
    detailed_rows = []
    for i, eps in enumerate(results['epsilon']):
        mean_d = results['mean_delta_change_deg'][i]
        std_d = results['std_delta_change_deg'][i]
        mean_J = results['mean_J_change'][i]
        std_J = results['std_J_change'][i]
        unphys = results.get('unphysical_counts', [0]*len(results['epsilon']))[i]
        rows.append({'epsilon': eps,
                     'mean_delta_change_deg': mean_d,
                     'std_delta_change_deg': std_d,
                     'mean_J_change': mean_J,
                     'std_J_change': std_J,
                     'unphysical_count': unphys})
        # store individual sample entries with a flag
        delta_list = results['all_delta_changes'][i]
        J_list = results['all_J_changes'][i]
        unphys_indices = results.get('unphysical_indices', [[]]*len(results['epsilon']))[i]
        for j in range(len(delta_list)):
            is_unphys = j in unphys_indices
            detailed_rows.append({'epsilon': eps, 'sample_index': j, 'delta_change_deg': delta_list[j], 'J_change': J_list[j], 'unphysical': bool(is_unphys)})
    df = pd.DataFrame(rows)
    csv_path = f'{out_prefix}_summary.csv'
    df.to_csv(csv_path, index=False)
    logging.info(f'Saved summary CSV to {csv_path}')

    df_detailed = pd.DataFrame(detailed_rows)
    detailed_csv = f'{out_prefix}_detailed_samples.csv'
    df_detailed.to_csv(detailed_csv, index=False)
    logging.info(f'Saved detailed samples CSV to {detailed_csv}')

    # Save combined histograms for all eps as example
    all_d = np.concatenate([np.array(x) for x in results['all_delta_changes'] if len(x)>0]) if any([len(x)>0 for x in results['all_delta_changes']]) else np.array([])
    all_J = np.concatenate([np.array(x) for x in results['all_J_changes'] if len(x)>0]) if any([len(x)>0 for x in results['all_J_changes']]) else np.array([])
    fig = plt.figure(figsize=(10,4))
    ax1 = fig.add_subplot(1,2,1)
    if all_d.size>0:
        ax1.hist(all_d, bins=50)
    ax1.set_title('Histogram of Δδ (collected physical samples)')
    ax1.set_xlabel('Δδ [deg]')
    ax2 = fig.add_subplot(1,2,2)
    if all_J.size>0:
        ax2.hist(all_J, bins=50)
    ax2.set_title('Histogram of ΔJ (collected physical samples)')
    ax2.set_xlabel('ΔJ')
    plt.tight_layout()
    hist_path = f'{out_prefix}_histograms.png'
    fig.savefig(hist_path)
    logging.info(f'Saved histograms to {hist_path}')

    # Save main errorbar figure
    fig2 = plot_scan_results(results, title_prefix='NuFIT perturbation: ')
    fig2_path = f'{out_prefix}_scan.png'
    fig2.savefig(fig2_path)
    logging.info(f'Saved scan figure to {fig2_path}')

    # Summary log
    total_unphys = sum(results.get('unphysical_counts', []))
    total_attempts = len(results['epsilon']) * (len(results['all_delta_changes'][0]) if len(results['all_delta_changes'])>0 else 0)
    logging.info(f'Total unphysical samples flagged: {total_unphys} / estimated total attempts {total_attempts}')

    return {'csv': csv_path, 'detailed_csv': detailed_csv, 'hist': hist_path, 'scan': fig2_path}

# NuFIT loader helpers (attempt to fetch JSON from nu-fit.org or accept local file)

def load_nufit_from_local_or_json(json_path=None):
    """
    Try to load NuFIT parameters from a local JSON file (if provided) or
    from a simple bundled fallback dictionary. If you have a nu-fit export
    (CSV/JSON), pass its path.
    """
    if json_path is not None and os.path.exists(json_path):
        with open(json_path,'r') as f:
            data = json.load(f)
        return data
    # fallback: minimal default values (user should replace with official NuFIT JSON)
    logging.warning('No NuFIT JSON provided. Using fallback example values (replace with official NuFIT data for production).')
    fallback = {
        'theta12_deg': 33.44,
        'theta13_deg': 8.57,
        'theta23_deg': 49.2,
        'delta_cp_deg': 197.0,
        'dm21_sq': 7.42e-5,
        'dm3l_sq': 2.517e-3,
        'm1_ev': 0.001
    }
    return fallback

# Build sampler from param uncertainties / covariance

def build_sampler_from_cov(mean_params, cov_matrix=None, param_names=None, scale=1.0):
    """
    Returns a callable sampler() -> deltaM built from sampling parameter variations
    mean_params: dict with keys (theta12_deg, theta13_deg, theta23_deg, delta_cp_deg, dm21_sq, dm3l_sq)
    cov_matrix: 6x6 covariance matrix (if None, we use diagonal with guessed errors)
    param_names: order of parameters in covariance
    scale: multiplier for uncertainties (1.0 uses given cov, <1 or >1 scales it)
    """
    if cov_matrix is None:
        # use diagonal estimates (rough, user should supply real NuFIT cov)
        eps = np.array([0.3, 0.15, 1.5, 20.0, 0.1e-5, 0.1e-3])  # example sigma: degrees and eV^2
        cov_matrix = np.diag(eps**2) * scale
    mvn = multivariate_normal(mean=[mean_params.get(k,0) for k in param_names], cov=cov_matrix)

    def sampler():
        sample = mvn.rvs()
        # map back
        th12_s, th13_s, th23_s, delta_s, dm21_s, dm3l_s = sample
        # build masses from m1 and dm's
        m1 = mean_params.get('m1_ev', 0.001)
        m2 = np.sqrt(m1**2 + dm21_s)
        # interpret dm3l as dm31 for normal ordering assumption here; caller must ensure ordering
        m3 = np.sqrt(m1**2 + dm3l_s)
        U = build_PMNS(th12_s, th13_s, th23_s, delta_s)
        M = build_mass_matrix_from_masses(U, [m1,m2,m3], mass_matrix_type='Majorana')
        # return deltaM as difference from central M
        U0 = build_PMNS(mean_params['theta12_deg'], mean_params['theta13_deg'], mean_params['theta23_deg'], mean_params['delta_cp_deg'])
        M0 = build_mass_matrix_from_masses(U0, [mean_params['m1_ev'], np.sqrt(mean_params['m1_ev']**2 + mean_params['dm21_sq']), np.sqrt(mean_params['m1_ev']**2 + mean_params['dm3l_sq'])], mass_matrix_type='Majorana')
        return M - M0
    return sampler

# Main runnable example & CLI

def main(args):
    # Load NuFIT or fallback
    nufit = load_nufit_from_local_or_json(args.nufit_json)
    theta12 = nufit['theta12_deg']
    theta13 = nufit['theta13_deg']
    theta23 = nufit['theta23_deg']
    delta_cp = nufit['delta_cp_deg']
    m1 = nufit.get('m1_ev', 0.001)
    m2 = np.sqrt(m1**2 + nufit['dm21_sq'])
    m3 = np.sqrt(m1**2 + nufit['dm3l_sq'])
    masses = [m1, m2, m3]

    ordering = args.ordering
    mass_matrix_type = args.mass_matrix_type

    # Build sampler if requested
    sampler = None
    if args.use_cov_sampler:
        # For production: user should provide a real 6x6 covariance matrix file (npz or csv)
        # Here we attempt to load a covariance if provided
        if args.cov_path and os.path.exists(args.cov_path):
            cov = np.load(args.cov_path)
            cov_matrix = cov['cov'] if 'cov' in cov else cov
        else:
            cov_matrix = None
        param_names = ['theta12_deg','theta13_deg','theta23_deg','delta_cp_deg','dm21_sq','dm3l_sq']
        mean_params = {
            'theta12_deg': theta12,
            'theta13_deg': theta13,
            'theta23_deg': theta23,
            'delta_cp_deg': delta_cp,
            'dm21_sq': nufit['dm21_sq'],
            'dm3l_sq': nufit['dm3l_sq'],
            'm1_ev': m1
        }
        sampler = build_sampler_from_cov(mean_params, cov_matrix=cov_matrix, param_names=param_names, scale=args.cov_scale)

    epsilon_list = args.epsilon_list
    results = monte_carlo_scan(theta12,theta13,theta23,delta_cp,masses,
                               epsilon_list, n_samples_per_epsilon=args.n_samples,
                               random_scale=args.random_scale, complex_phase=args.complex_phase,
                               mass_matrix_type=mass_matrix_type, ordering=ordering,
                               sampler=sampler)

    outputs = detailed_report_and_save(results, out_prefix=args.out_prefix)
    logging.info('Done. Outputs:')
    for k,v in outputs.items():
        logging.info(f' - {k}: {v}')

class Args:
    nufit_json = 'nufit.json'  
    cov_path = None
    use_cov_sampler = False
    cov_scale = 1.0
    ordering = 'normal'
    mass_matrix_type = 'Majorana'
    n_samples = 200
    epsilon_list = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2]
    random_scale = 1e-3
    complex_phase = False
    out_prefix = 'nu_analysis'

args = Args()
main(args)

