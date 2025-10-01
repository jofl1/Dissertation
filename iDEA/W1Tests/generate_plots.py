#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
import iDEA

# Create output directory for plots
import os
if not os.path.exists('W1Tests'):
    os.makedirs('W1Tests')

print("Generating plots for iDEA project book...\n")

# ==================== Plot 1: Two-Electron Atom Ground State ====================
print("1. Computing two-electron atom ground state...")
system_atom = iDEA.system.systems.atom
ground_state = iDEA.methods.interacting.solve(system_atom, k=0)
n = iDEA.observables.density(system_atom, state=ground_state)

plt.figure(figsize=(10, 6))
plt.plot(system_atom.x, n, 'b-', linewidth=2, label='Electron density')
plt.xlabel('Position x (a.u.)', fontsize=12)
plt.ylabel('Density n(x) (a.u.)', fontsize=12)
plt.title('Ground State Electron Density for Two-Electron Atom', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('W1Tests/1_two_electron_atom_density.png', dpi=300)
print(f"   Total Energy: {ground_state.energy:.6f} a.u.")
print(f"   Plot saved: W1Tests/1_two_electron_atom_density.png\n")

# ==================== Plot 2: Harmonic Oscillator System ====================
print("2. Computing harmonic oscillator ground state...")
x = np.linspace(-10, 10, 500)
v_ext = 0.5 * 0.25**2 * x**2  # omega = 0.25
v_int = iDEA.interactions.softened_interaction(x)
system_ho = iDEA.system.System(x, v_ext, v_int, electrons='ud')

ground_state_ho = iDEA.methods.interacting.solve(system_ho, k=0)
n_ho = iDEA.observables.density(system_ho, state=ground_state_ho)

plt.figure(figsize=(10, 6))
plt.plot(x, n_ho, 'r-', linewidth=2, label='Electron density')
plt.plot(x, v_ext, 'g--', linewidth=1.5, label='External potential', alpha=0.7)
plt.xlabel('Position x (a.u.)', fontsize=12)
plt.ylabel('Density / Potential (a.u.)', fontsize=12)
plt.title('Harmonic Oscillator: Ground State Density and Potential', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('W1Tests/2_harmonic_oscillator.png', dpi=300)
print(f"   Total Energy: {ground_state_ho.energy:.6f} a.u.")
print(f"   Plot saved: W1Tests/2_harmonic_oscillator.png\n")

# ==================== Plot 3: Excited States Comparison ====================
print("3. Computing excited states (k=0,1,2)...")
x = np.linspace(-8, 8, 400)
v_ext = 0.5 * 0.5**2 * x**2
v_int = iDEA.interactions.softened_interaction(x)
system_exc = iDEA.system.System(x, v_ext, v_int, electrons='u')

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
energies = []

for k in range(3):
    state = iDEA.methods.interacting.solve(system_exc, k=k)
    n_exc = iDEA.observables.density(system_exc, state=state)
    energies.append(state.energy)

    axes[k].plot(x, n_exc, 'b-', linewidth=2)
    axes[k].set_xlabel('Position x (a.u.)', fontsize=11)
    axes[k].set_ylabel('Density n(x) (a.u.)', fontsize=11)
    axes[k].set_title(f'State k={k}\nE = {state.energy:.4f} a.u.', fontsize=12, fontweight='bold')
    axes[k].grid(True, alpha=0.3)

plt.suptitle('Excited States of Quantum Harmonic Oscillator', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('W1Tests/3_excited_states.png', dpi=300)
print(f"   Energies: k=0: {energies[0]:.4f}, k=1: {energies[1]:.4f}, k=2: {energies[2]:.4f} a.u.")
print(f"   Plot saved: W1Tests/3_excited_states.png\n")

# ==================== Plot 4: Many-Body vs Non-Interacting Comparison ====================
print("4. Comparing interacting vs non-interacting electrons...")
system_comp = iDEA.system.systems.atom

# Interacting solution
state_int = iDEA.methods.interacting.solve(system_comp, k=0)
n_int = iDEA.observables.density(system_comp, state=state_int)

# Non-interacting solution
state_non = iDEA.methods.non_interacting.solve(system_comp, k=0)
n_non = iDEA.observables.density(system_comp, state=state_non)

# Get total energy for non-interacting system
E_non = iDEA.observables.single_particle_energy(system_comp, state=state_non)

plt.figure(figsize=(10, 6))
plt.plot(system_comp.x, n_int, 'b-', linewidth=2, label='Interacting electrons')
plt.plot(system_comp.x, n_non, 'r--', linewidth=2, label='Non-interacting electrons')
plt.xlabel('Position x (a.u.)', fontsize=12)
plt.ylabel('Density n(x) (a.u.)', fontsize=12)
plt.title('Electron-Electron Interaction Effects on Density', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.legend(fontsize=11)
plt.tight_layout()
plt.savefig('W1Tests/4_interaction_comparison.png', dpi=300)
print(f"   Interacting Energy: {state_int.energy:.6f} a.u.")
print(f"   Non-interacting Energy: {E_non:.6f} a.u.")
print(f"   Correlation Energy: {state_int.energy - E_non:.6f} a.u.")
print(f"   Plot saved: W1Tests/4_interaction_comparison.png\n")

