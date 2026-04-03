"""
Test: Vortex-induced vibration — CPU vs MetalHLO comparison.
Runs reduced domain/timesteps and saves displacement + force history.
"""

import os
import math

backend = os.environ.get("METALHLO_TEST_BACKEND", "cpu")
output_dir = os.environ.get("METALHLO_TEST_OUTPUT_DIR", "/tmp/test_viv")
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
D = 16  # Reduced from 32
NX = 10 * D
NY = 5 * D
CYL_X = 3 * D
CYL_Y = 2.5 * D  # Use float to keep centered
CYL_Y = NY // 2  # Integer center
CYL_AREA = math.pi * (D / 2) ** 2

N_MARKER = 4 * D
MARKER_THETA = jnp.linspace(0, 2 * jnp.pi, N_MARKER, endpoint=False)
MARKER_X = CYL_X + 0.5 * D * jnp.cos(MARKER_THETA)
MARKER_Y = CYL_Y + 0.5 * D * jnp.sin(MARKER_THETA)
MARKER_DS = 2 * math.pi * (D / 2) / N_MARKER

U0 = 0.05
RE = 150
UR = 5
MR = 10
DR = 0

NU = U0 * D / RE
FN = U0 / (UR * D)
M = CYL_AREA * MR
K = (2 * math.pi * FN) ** 2 * M * (1 + 1 / MR)
C = 2 * math.sqrt(K * M) * DR

TM = 1500  # Reduced from ~48000

OMEGA = lbm.get_omega(NU)
IB_ITER = 1
IB_PAD = 10
FSI_ITER = 1

IB_X0 = int(CYL_X - 0.5 * D - IB_PAD)
IB_Y0 = int(CYL_Y - 0.5 * D - IB_PAD)
IB_SIZE = D + 2 * IB_PAD

# Initialize
rho = jnp.ones((NX, NY))
u = jnp.zeros((2, NX, NY))
u = u.at[0].set(U0)
f = lbm.get_equilibrium(rho, u)

d = jnp.zeros(2)
v = jnp.zeros(2)
a = jnp.zeros(2)
v = v.at[1].set(1e-2 * U0)

@jax.jit
def update(f, d, v, a):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)

    ib_x0 = (IB_X0 + d[0]).astype(jnp.int32)
    ib_y0 = (IB_Y0 + d[1]).astype(jnp.int32)

    ib_rho = jax.lax.dynamic_slice(rho, (ib_x0, ib_y0), (IB_SIZE, IB_SIZE))
    ib_u = jax.lax.dynamic_slice(u, (0, ib_x0, ib_y0), (2, IB_SIZE, IB_SIZE))
    ib_f = jax.lax.dynamic_slice(f, (0, ib_x0, ib_y0), (9, IB_SIZE, IB_SIZE))

    a_old, v_old, d_old = a, v, d

    for _ in range(FSI_ITER):
        marker_x, marker_y = dyn.get_markers_coords_2dof(MARKER_X, MARKER_Y, d)

        stencil_weights, stencil_indices = ib.get_ib_stencil(
            marker_x=marker_x - ib_x0,
            marker_y=marker_y - ib_y0,
            ny=IB_SIZE,
            kernel=ib.kernel_peskin_4pt,
            stencil_radius=2,
        )
        marker_v = jnp.repeat(v[None, :], N_MARKER, axis=0)

        ib_g, marker_h = ib.multi_direct_forcing(
            grid_u=ib_u,
            stencil_weights=stencil_weights,
            stencil_indices=stencil_indices,
            marker_u_target=marker_v,
            marker_ds=MARKER_DS,
            n_iter=IB_ITER,
        )

        h = dyn.get_force_to_obj(marker_h)
        h += a * CYL_AREA
        a, v, d = dyn.newmark_2dof(a_old, v_old, d_old, h, M, K, C)

    ib_f = lbm.forcing_edm(ib_f, ib_g, ib_u, ib_rho)
    f = jax.lax.dynamic_update_slice(f, ib_f, (0, ib_x0, ib_y0))

    f = lbm.streaming(f)
    f = lbm.boundary_nebb(f, loc="left", ux_wall=U0)
    f = lbm.boundary_equilibrium(f, loc="right", ux_wall=U0)

    return f, d, v, a, h

# Run simulation
d_hist = np.zeros((2, TM), dtype=np.float32)
h_hist = np.zeros((2, TM), dtype=np.float32)

for t in range(TM):
    f, d, v, a, h = update(f, d, v, a)
    d_hist[:, t] = np.asarray(d)
    h_hist[:, t] = np.asarray(h)
    if t % 300 == 0:
        print(f"  step {t}/{TM}  d=[{float(d[0]):.6f}, {float(d[1]):.6f}]")

# Save results
np.save(os.path.join(output_dir, "f.npy"), np.array(f))
np.save(os.path.join(output_dir, "d_hist.npy"), d_hist)
np.save(os.path.join(output_dir, "h_hist.npy"), h_hist)
np.save(os.path.join(output_dir, "d_final.npy"), np.array(d))
np.save(os.path.join(output_dir, "v_final.npy"), np.array(v))

cd = h_hist[0] * 2 / (D * U0 ** 2)
cl = h_hist[1] * 2 / (D * U0 ** 2)
np.save(os.path.join(output_dir, "cd.npy"), cd)
np.save(os.path.join(output_dir, "cl.npy"), cl)

print(f"Saved results to {output_dir}")
print(f"  d_final: [{float(d[0]):.8f}, {float(d[1]):.8f}]")
print(f"  cd range: [{cd.min():.6f}, {cd.max():.6f}]")
print(f"  cl range: [{cl.min():.6f}, {cl.max():.6f}]")
