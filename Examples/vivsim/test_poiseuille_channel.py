"""
Test: Poiseuille channel flow — CPU vs MetalHLO comparison.
Runs BGK and KBC collision/forcing combos and saves velocity profiles.
Note: MRT recipe is skipped because vivsim computes MRT constants at import
time using jnp.linalg.inv, which requires while loops not yet supported.
"""

import os

backend = os.environ.get("METALHLO_TEST_BACKEND", "cpu")
output_dir = os.environ.get("METALHLO_TEST_OUTPUT_DIR", "/tmp/test_poiseuille_channel")
os.makedirs(output_dir, exist_ok=True)

if backend == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["JAX_PLATFORMS"] = "metalhlo,cpu"

import jax
import jax.numpy as jnp
import numpy as np
from vivsim import lbm

print(f"Backend: {backend}")
print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

NU = 0.2
GX = 0.001
NX, NY = 10, 10
OMEGA = lbm.get_omega(NU)
TM = 20000  # Reduced from 100000

UX_WALL, _ = lbm.get_corrected_wall_velocity(0, 0, gx_wall=GX)


def run_recipe(name, update_fn, tm=TM):
    """Run a recipe and return final state."""
    rho = jnp.ones((NX, NY))
    u = jnp.zeros((2, NX, NY))
    g = jnp.zeros((2, NX, NY))
    g = g.at[0].set(GX)
    f = lbm.get_equilibrium(rho, u - g / 2)

    for i in range(tm):
        f, rho, u = update_fn(f)
        if i % 5000 == 0:
            print(f"  {name}: step {i}/{tm}")

    return np.array(f), np.array(rho), np.array(u)


# Global force field
g_global = jnp.zeros((2, NX, NY))
g_global = g_global.at[0].set(GX)

# --- BGK + EDM ---
@jax.jit
def main_bgk_edm(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.forcing_edm(f, g_global, u, rho)
    u = u + lbm.get_velocity_correction(g_global, rho)
    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc='top', ux_wall=UX_WALL)
    f = lbm.boundary_nebb(f, loc='bottom', ux_wall=UX_WALL)
    return f, rho, u

# --- BGK + Guo ---
@jax.jit
def main_bgk_guo(f):
    rho, u = lbm.get_macroscopic(f)
    u = u + lbm.get_velocity_correction(g_global, rho)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_bgk(f, feq, OMEGA)
    f = lbm.forcing_guo_bgk(f, g_global, u, OMEGA)
    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc='top', ux_wall=UX_WALL)
    f = lbm.boundary_nebb(f, loc='bottom', ux_wall=UX_WALL)
    return f, rho, u

# --- KBC + EDM ---
@jax.jit
def main_kbc_edm(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)
    f = lbm.forcing_edm(f, g_global, u, rho)
    u += lbm.get_velocity_correction(g_global, rho)
    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc='top', ux_wall=UX_WALL)
    f = lbm.boundary_nebb(f, loc='bottom', ux_wall=UX_WALL)
    return f, rho, u


recipes = [
    ("bgk_edm", main_bgk_edm),
    ("bgk_guo", main_bgk_guo),
    ("kbc_edm", main_kbc_edm),
]

for name, fn in recipes:
    print(f"\nRunning {name}...")
    f_np, rho_np, u_np = run_recipe(name, fn)

    np.save(os.path.join(output_dir, f"f_{name}.npy"), f_np)
    np.save(os.path.join(output_dir, f"rho_{name}.npy"), rho_np)
    np.save(os.path.join(output_dir, f"u_{name}.npy"), u_np)

    u_centerline = u_np[0, NX // 2, :]
    np.save(os.path.join(output_dir, f"u_centerline_{name}.npy"), u_centerline)
    print(f"  u_centerline range: [{u_centerline.min():.8f}, {u_centerline.max():.8f}]")

# Analytical solution for reference
H = NY - 1
y = np.arange(NY, dtype=np.float32)
ux_analytical = GX / 2 / NU * (y * (H - y))
np.save(os.path.join(output_dir, "u_analytical.npy"), ux_analytical)

print(f"\nSaved results to {output_dir}")
