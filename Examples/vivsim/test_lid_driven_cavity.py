"""
Test: 2D lid-driven cavity flow — CPU vs MetalHLO comparison.
Runs a reduced number of timesteps and saves final velocity/density fields.
"""

import os
import sys

backend = os.environ.get("METALHLO_TEST_BACKEND", "cpu")
output_dir = os.environ.get("METALHLO_TEST_OUTPUT_DIR", "/tmp/test_lid_driven_cavity")
os.makedirs(output_dir, exist_ok=True)

if backend == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["JAX_PLATFORMS"] = "metalhlo,cpu"

import jax
import jax.numpy as jnp
import numpy as np
from vivsim import lbm, post

print(f"Backend: {backend}")
print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Parameters
U0 = 0.3
RE_GRID = 30
NU = U0 / RE_GRID
OMEGA = lbm.get_omega(NU)

NX = 100  # Reduced from 1000
NY = 100
TM = 2000  # Reduced from 80000

rho = jnp.ones((NX, NY))
u = jnp.zeros((2, NX, NY))
f = lbm.get_equilibrium(rho, u)

@jax.jit
def update(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.streaming(f)
    f = lbm.boundary_nee(f, loc='top', ux_wall=U0)
    f = lbm.boundary_nee(f, loc='left')
    f = lbm.boundary_nee(f, loc='right')
    f = lbm.boundary_nee(f, loc='bottom')
    return f, rho, u

# Run simulation
for t in range(TM):
    f, rho, u = update(f)
    if t % 500 == 0:
        print(f"  step {t}/{TM}")

# Force results to host
f_np = np.array(f)
rho_np = np.array(rho)
u_np = np.array(u)
vel_mag = np.array(post.calculate_velocity_magnitude(u))

# Save results
np.save(os.path.join(output_dir, "f.npy"), f_np)
np.save(os.path.join(output_dir, "rho.npy"), rho_np)
np.save(os.path.join(output_dir, "u.npy"), u_np)
np.save(os.path.join(output_dir, "vel_magnitude.npy"), vel_mag)

print(f"Saved results to {output_dir}")
print(f"  rho range: [{rho_np.min():.6f}, {rho_np.max():.6f}]")
print(f"  u range: [{u_np.min():.6f}, {u_np.max():.6f}]")
print(f"  vel_mag range: [{vel_mag.min():.6f}, {vel_mag.max():.6f}]")
