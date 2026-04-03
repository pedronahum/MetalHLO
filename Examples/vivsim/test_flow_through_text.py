"""
Test: Flow through text — CPU vs MetalHLO comparison.
Runs reduced domain/timesteps and saves final fields.
"""

import os

backend = os.environ.get("METALHLO_TEST_BACKEND", "cpu")
output_dir = os.environ.get("METALHLO_TEST_OUTPUT_DIR", "/tmp/test_flow_through_text")
os.makedirs(output_dir, exist_ok=True)

if backend == "cpu":
    os.environ["JAX_PLATFORMS"] = "cpu"
else:
    os.environ["JAX_PLATFORMS"] = "metalhlo,cpu"

import jax
import jax.numpy as jnp
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from vivsim import lbm, post

print(f"Backend: {backend}")
print(f"JAX default backend: {jax.default_backend()}")
print(f"JAX devices: {jax.devices()}")

# Parameters (reduced)
U0 = 0.05
RE_GRID = 20
NU = U0 / RE_GRID
OMEGA = lbm.get_omega(NU)

NX = 100  # Reduced from 500
NY = 100
TM = 2000  # Reduced from 80000

rho = jnp.ones((NX, NY), dtype=jnp.float32)
u = jnp.zeros((2, NX, NY), dtype=jnp.float32)

# Generate text mask
TEXT = 'HLO'  # Shorter text for smaller domain
FONT_SIZE = NX // 5

img = Image.new("L", (NX, NY), 0)
draw = ImageDraw.Draw(img)
try:
    font = ImageFont.truetype('DejaVuSans.ttf', FONT_SIZE)
except OSError:
    # Fallback: use default font
    font = ImageFont.load_default()

draw.text((NX // 2, NY // 2), TEXT, font=font, fill=255, anchor='mm')
MASK = jnp.array(img).astype(bool)[::-1].T

# Save mask for comparison (should be identical across backends)
np.save(os.path.join(output_dir, "mask.npy"), np.array(MASK))

u = u.at[1].set(U0)
f = lbm.get_equilibrium(rho, u)

@jax.jit
def update(f):
    rho, u = lbm.get_macroscopic(f)
    feq = lbm.get_equilibrium(rho, u)
    f = lbm.collision_kbc(f, feq, OMEGA)
    f = lbm.streaming(f)
    f = lbm.boundary_nee(f, loc='bottom', uy_wall=U0)
    f = lbm.boundary_equilibrium(f, loc='top', uy_wall=U0)
    f = lbm.obstacle_bounce_back(f, MASK)
    return f, rho, u

for t in range(TM):
    f, rho, u = update(f)
    if t % 500 == 0:
        print(f"  step {t}/{TM}")

# Save results
f_np = np.array(f)
rho_np = np.array(rho)
u_np = np.array(u)
vel_mag = np.array(post.calculate_velocity_magnitude(u))

np.save(os.path.join(output_dir, "f.npy"), f_np)
np.save(os.path.join(output_dir, "rho.npy"), rho_np)
np.save(os.path.join(output_dir, "u.npy"), u_np)
np.save(os.path.join(output_dir, "vel_magnitude.npy"), vel_mag)

print(f"Saved results to {output_dir}")
print(f"  rho range: [{rho_np.min():.6f}, {rho_np.max():.6f}]")
print(f"  vel_mag range: [{vel_mag.min():.6f}, {vel_mag.max():.6f}]")
