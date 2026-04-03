"""
Test: Flow past cylinder — CPU vs MetalHLO comparison.
Runs reduced domain/timesteps and saves force history + final fields.
"""

import os
import math

backend = os.environ.get("METALHLO_TEST_BACKEND", "cpu")
output_dir = os.environ.get("METALHLO_TEST_OUTPUT_DIR", "/tmp/test_flow_pass_cylinder")
os.makedirs(output_dir, exist_ok=True)

if backend == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["JAX_PLATFORMS"] = "metalhlo,cpu"

import jax
import jax.numpy as jnp
import numpy as np
from vivsim import dyn, ib, lbm

print(f"Backend: {backend}")
print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Reduced parameters
D = 20  # Reduced from 60
NX = 15 * D
NY = 10 * D
CYL_X = 5 * D
CYL_Y = 5 * D
CYL_AREA = math.pi * (D / 2) ** 2

N_MARKER = 4 * D
MARKER_THETA = jnp.linspace(0, 2 * jnp.pi, N_MARKER, endpoint=False)
MARKER_X = CYL_X + 0.5 * D * jnp.cos(MARKER_THETA)
MARKER_Y = CYL_Y + 0.5 * D * jnp.sin(MARKER_THETA)
MARKER_COORDS = jnp.stack((MARKER_X, MARKER_Y), axis=1)
MARKER_DS = ib.get_ds_closed(MARKER_COORDS)

U0 = 0.05
RE = 200
NU = U0 * D / RE
TM = 3000  # Reduced from ~180000

OMEGA = lbm.get_omega(NU)
IB_ITER = 3
IB_PAD = 2

IB_X0 = int(CYL_X - 0.5 * D - IB_PAD)
IB_Y0 = int(CYL_Y - 0.5 * D - IB_PAD)
IB_SIZE = D + 2 * IB_PAD

MARKER_X_IB = MARKER_X - IB_X0
MARKER_Y_IB = MARKER_Y - IB_Y0
STENCIL_WEIGHTS, STENCIL_INDICES = ib.get_ib_stencil(
    MARKER_X_IB, MARKER_Y_IB, IB_SIZE, kernel=ib.kernel_peskin_4pt,
)

rho = jnp.ones((NX, NY))
u = jnp.zeros((2, NX, NY))
u = u.at[0].set(U0)
f = lbm.get_equilibrium(rho, u)

marker_v = jnp.zeros((N_MARKER, 2))

@jax.jit
def update(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)

    ib_rho = jax.lax.dynamic_slice(rho, (IB_X0, IB_Y0), (IB_SIZE, IB_SIZE))
    ib_u = jax.lax.dynamic_slice(u, (0, IB_X0, IB_Y0), (2, IB_SIZE, IB_SIZE))
    ib_f = jax.lax.dynamic_slice(f, (0, IB_X0, IB_Y0), (9, IB_SIZE, IB_SIZE))

    ib_g, marker_h = ib.multi_direct_forcing(
        grid_u=ib_u,
        stencil_weights=STENCIL_WEIGHTS,
        stencil_indices=STENCIL_INDICES,
        marker_u_target=marker_v,
        marker_ds=MARKER_DS,
        n_iter=IB_ITER,
    )

    h = dyn.get_force_to_obj(marker_h)

    ib_f = lbm.forcing_edm(ib_f, ib_g, ib_u, ib_rho)
    f = jax.lax.dynamic_update_slice(f, ib_f, (0, IB_X0, IB_Y0))

    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc="left", ux_wall=U0)
    f = lbm.boundary_equilibrium(f, loc="right", ux_wall=U0)

    return f, h

# Run simulation and record force history
h_hist = np.zeros((2, TM), dtype=np.float32)
for t in range(TM):
    f, h = update(f)
    h_hist[:, t] = np.asarray(h)
    if t % 500 == 0:
        print(f"  step {t}/{TM}")

# Final macroscopic fields
rho_final, u_final = lbm.get_macroscopic(f)

np.save(os.path.join(output_dir, "f.npy"), np.array(f))
np.save(os.path.join(output_dir, "rho.npy"), np.array(rho_final))
np.save(os.path.join(output_dir, "u.npy"), np.array(u_final))
np.save(os.path.join(output_dir, "h_hist.npy"), h_hist)

cd = h_hist[0] * 2 / (D * U0 ** 2)
cl = h_hist[1] * 2 / (D * U0 ** 2)
np.save(os.path.join(output_dir, "cd.npy"), cd)
np.save(os.path.join(output_dir, "cl.npy"), cl)

print(f"Saved results to {output_dir}")
print(f"  cd range: [{cd.min():.6f}, {cd.max():.6f}]")
print(f"  cl range: [{cl.min():.6f}, {cl.max():.6f}]")
