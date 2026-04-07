#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
import argparse
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage.measure import marching_cubes


BDF_MAX = 0.05
ISO_LEVEL_MAX_FRAC_OF_BDF = 0.8
DEFAULT_DEVICE = "cuda"
BATCH_SIZE = 131072

Q_FALLBACK_IN_DIM = 8
Q_FALLBACK_HIDDEN = 128
Q_FALLBACK_LAYERS = 3

GRID_RESOLUTION = 100


# ============================================================
# model
# ============================================================

class SimpleDecoder(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int = 512, num_layers: int = 8):
        super().__init__()
        layers = []
        last = in_dim
        for _ in range(num_layers):
            layers += [nn.Linear(last, hidden_dim), nn.ReLU(inplace=True)]
            last = hidden_dim
        self.net = nn.Sequential(*layers)
        self.head = nn.Linear(last, 1)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.net(torch.cat([z, x], dim=-1)))


def load_model(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(str(ckpt_path), map_location=device)
    z_sdf = ckpt["z_sdf"].to(device=device).float()
    z_bdf = ckpt["z_udf"].to(device=device).float()
    latent_dim = int(z_sdf.shape[0])

    hparams = ckpt.get("hparams", {})
    hidden_dim = int(hparams.get("hidden_dim", 512))
    num_layers = int(hparams.get("num_layers", 8))

    dec_sdf = SimpleDecoder(latent_dim + 3, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    dec_bdf = SimpleDecoder(latent_dim + 3, hidden_dim=hidden_dim, num_layers=num_layers).to(device)
    dec_sdf.load_state_dict(ckpt["dec_sdf"])
    dec_bdf.load_state_dict(ckpt["dec_udf"])
    dec_sdf.eval()
    dec_bdf.eval()
    return dec_sdf, dec_bdf, z_sdf, z_bdf


class QGateMLP(nn.Module):
    def __init__(self, in_dim: int = 8, hidden: int = 128, layers: int = 3):
        super().__init__()
        net = []
        d = in_dim
        for _ in range(layers):
            net += [nn.Linear(d, hidden), nn.SiLU(inplace=True)]
            d = hidden
        net += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*net)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(feat)).squeeze(-1)


def load_q_mlp(
    q_ckpt_path: Path,
    device: torch.device,
    fallback_in_dim: int,
    fallback_hidden: int,
    fallback_layers: int,
):
    obj = torch.load(str(q_ckpt_path), map_location=device)

    hparams = {}
    state = None

    if isinstance(obj, dict):
        hparams = obj.get("hparams", {}) if isinstance(obj.get("hparams", {}), dict) else {}
        if "q_mlp" in obj:
            state = obj["q_mlp"]
        elif "state_dict" in obj:
            state = obj["state_dict"]
        else:
            if any(isinstance(k, str) and ("weight" in k or "bias" in k) for k in obj.keys()):
                state = obj

    if state is None:
        raise ValueError(f"Unrecognized q-ckpt format: {q_ckpt_path}")

    in_dim = int(hparams.get("in_dim", fallback_in_dim))
    hidden = int(hparams.get("hidden", fallback_hidden))
    layers = int(hparams.get("layers", fallback_layers))

    q_mlp = QGateMLP(in_dim=in_dim, hidden=hidden, layers=layers).to(device)
    q_mlp.load_state_dict(state, strict=True)
    q_mlp.eval()
    return q_mlp


# ============================================================
# geometry utils
# ============================================================

def _normalize_axis(axis: np.ndarray, name: str) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float32).reshape(3)
    n = float(np.linalg.norm(axis))
    if not np.isfinite(n) or n < 1e-12:
        raise ValueError(f"{name} must be a non-zero finite 3D vector, got {axis}")
    return axis / n


def make_R_axis_angle(rot_deg: float, axis: np.ndarray) -> np.ndarray:
    a = _normalize_axis(axis, "axis")
    theta = np.deg2rad(rot_deg)
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    x, y, z = float(a[0]), float(a[1]), float(a[2])

    return np.array([
        [c + x * x * (1 - c),       x * y * (1 - c) - z * s,   x * z * (1 - c) + y * s],
        [y * x * (1 - c) + z * s,   c + y * y * (1 - c),       y * z * (1 - c) - x * s],
        [z * x * (1 - c) - y * s,   z * y * (1 - c) + x * s,   c + z * z * (1 - c)],
    ], dtype=np.float32)


def world_to_canonical_torch(
    x_world: torch.Tensor,
    pivot: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    return (x_world - pivot.unsqueeze(0)) @ R + pivot.unsqueeze(0)


def build_X_new_qblend(sdf: torch.Tensor, bdf: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    return q * sdf - (1.0 - q) * bdf


# ============================================================
# coupled fields
# ============================================================

def obj_sdf_world(
    dec_sdf: nn.Module,
    z_sdf: torch.Tensor,
    x_world: torch.Tensor,
    kind: int,
    pivot: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    x_in = x_world if kind == 1 else world_to_canonical_torch(x_world, pivot=pivot, R=R)
    z = z_sdf.unsqueeze(0).expand(x_world.shape[0], -1)
    return dec_sdf(z, x_in).squeeze(1)


def obj_bdf_world(
    dec_bdf: nn.Module,
    z_bdf: torch.Tensor,
    x_world: torch.Tensor,
    kind: int,
    pivot: torch.Tensor,
    R: torch.Tensor,
) -> torch.Tensor:
    x_in = x_world if kind == 1 else world_to_canonical_torch(x_world, pivot=pivot, R=R)
    z = z_bdf.unsqueeze(0).expand(x_world.shape[0], -1)
    return F.relu(dec_bdf(z, x_in).squeeze(1))


def coupled_sdf_bdf_functional(
    x_world: torch.Tensor,
    dec1_sdf: nn.Module, z1_sdf: torch.Tensor,
    dec2_sdf: nn.Module, z2_sdf: torch.Tensor,
    dec3_sdf: nn.Module, z3_sdf: torch.Tensor,
    dec1_bdf: nn.Module, z1_bdf: torch.Tensor,
    dec2_bdf: nn.Module, z2_bdf: torch.Tensor,
    dec3_bdf: nn.Module, z3_bdf: torch.Tensor,
    pivot2: torch.Tensor, R2: torch.Tensor,
    pivot3: torch.Tensor, R3: torch.Tensor,
):
    s1 = obj_sdf_world(dec1_sdf, z1_sdf, x_world, kind=1, pivot=pivot2, R=R2)
    s2 = obj_sdf_world(dec2_sdf, z2_sdf, x_world, kind=2, pivot=pivot2, R=R2)
    s3 = obj_sdf_world(dec3_sdf, z3_sdf, x_world, kind=3, pivot=pivot3, R=R3)

    b1 = obj_bdf_world(dec1_bdf, z1_bdf, x_world, kind=1, pivot=pivot2, R=R2)
    b2 = obj_bdf_world(dec2_bdf, z2_bdf, x_world, kind=2, pivot=pivot2, R=R2)
    b3 = obj_bdf_world(dec3_bdf, z3_bdf, x_world, kind=3, pivot=pivot3, R=R3)

    sdf_c = torch.min(torch.stack([s1, s2, s3], dim=0), dim=0).values
    bdf_c = torch.min(torch.stack([b1, b2, b3], dim=0), dim=0).values
    return sdf_c, bdf_c


def apply_bdf_validity_to_xpred(
    sdf_c: torch.Tensor,
    bdf_c: torch.Tensor,
    q: torch.Tensor,
    bdf_max: float = BDF_MAX,
):
    valid_mask = bdf_c <= float(bdf_max)

    pos_inf = torch.full_like(sdf_c, float("inf"))
    neg_inf = torch.full_like(sdf_c, float("-inf"))
    x_pred = torch.where(sdf_c >= 0.0, pos_inf, neg_inf)

    if valid_mask.any():
        x_pred[valid_mask] = build_X_new_qblend(sdf_c[valid_mask], bdf_c[valid_mask], q[valid_mask])

    return x_pred


# ============================================================
# mesh utils
# ============================================================

def save_obj(path: str, verts: np.ndarray, faces: np.ndarray):
    with open(path, "w", encoding="utf-8") as f:
        for v in verts:
            f.write(f"v {v[0]:.8f} {v[1]:.8f} {v[2]:.8f}\n")
        for tri in faces:
            f.write(f"f {tri[0] + 1} {tri[1] + 1} {tri[2] + 1}\n")


def iso_surface_from_field(
    xs: np.ndarray,
    ys: np.ndarray,
    zs: np.ndarray,
    field: np.ndarray,
    level: float,
):
    finite_mask = np.isfinite(field)
    if not np.any(finite_mask):
        return None, None

    finite_vals = field[finite_mask]
    vmin = float(finite_vals.min())
    vmax = float(finite_vals.max())
    if not (vmin <= level <= vmax):
        return None, None

    safe_field = field.copy()
    safe_hi = float(vmax + max(1.0, abs(vmax) + abs(level) + 1.0))
    safe_lo = float(vmin - max(1.0, abs(vmin) + abs(level) + 1.0))
    safe_field[np.isposinf(safe_field)] = safe_hi
    safe_field[np.isneginf(safe_field)] = safe_lo

    verts_idx, faces, _, _ = marching_cubes(
        volume=safe_field,
        level=level,
        spacing=(1.0, 1.0, 1.0),
    )

    Nx, Ny, Nz = len(xs), len(ys), len(zs)
    sx = (xs[-1] - xs[0]) / (Nx - 1) if Nx > 1 else 1.0
    sy = (ys[-1] - ys[0]) / (Ny - 1) if Ny > 1 else 1.0
    sz = (zs[-1] - zs[0]) / (Nz - 1) if Nz > 1 else 1.0

    verts_world = np.empty_like(verts_idx)
    verts_world[:, 0] = xs[0] + verts_idx[:, 0] * sx
    verts_world[:, 1] = ys[0] + verts_idx[:, 1] * sy
    verts_world[:, 2] = zs[0] + verts_idx[:, 2] * sz

    return verts_world.astype(np.float32), faces.astype(np.int32)


# ============================================================
# grid evaluation
# ============================================================

def evaluate_q_field_on_grid(
    grid_min: np.ndarray,
    grid_max: np.ndarray,
    res: int,
    batch_size: int,
    device: torch.device,
    q_mlp: nn.Module,
    dec1_sdf: nn.Module, z1_sdf: torch.Tensor,
    dec2_sdf: nn.Module, z2_sdf: torch.Tensor,
    dec3_sdf: nn.Module, z3_sdf: torch.Tensor,
    dec1_bdf: nn.Module, z1_bdf: torch.Tensor,
    dec2_bdf: nn.Module, z2_bdf: torch.Tensor,
    dec3_bdf: nn.Module, z3_bdf: torch.Tensor,
    pivot2_t: torch.Tensor, R2_t: torch.Tensor,
    pivot3_t: torch.Tensor, R3_t: torch.Tensor,
):
    xs = np.linspace(grid_min[0], grid_max[0], res, dtype=np.float32)
    ys = np.linspace(grid_min[1], grid_max[1], res, dtype=np.float32)
    zs = np.linspace(grid_min[2], grid_max[2], res, dtype=np.float32)

    X, Y, Z = np.meshgrid(xs, ys, zs, indexing="ij")
    coords = np.stack([X, Y, Z], axis=-1).reshape(-1, 3).astype(np.float32)

    N = coords.shape[0]
    values = np.empty((N,), dtype=np.float32)

    start = 0
    while start < N:
        end = min(start + batch_size, N)
        x = torch.from_numpy(coords[start:end]).to(device=device, dtype=torch.float32)
        x_req = x.detach().requires_grad_(True)

        sdf_c, bdf_c = coupled_sdf_bdf_functional(
            x_req,
            dec1_sdf, z1_sdf, dec2_sdf, z2_sdf, dec3_sdf, z3_sdf,
            dec1_bdf, z1_bdf, dec2_bdf, z2_bdf, dec3_bdf, z3_bdf,
            pivot2_t, R2_t, pivot3_t, R3_t,
        )

        grad_sdf = torch.autograd.grad(
            outputs=sdf_c.sum(),
            inputs=x_req,
            create_graph=False,
            retain_graph=True,
            only_inputs=True,
        )[0]

        grad_bdf = torch.autograd.grad(
            outputs=bdf_c.sum(),
            inputs=x_req,
            create_graph=False,
            retain_graph=False,
            only_inputs=True,
        )[0]

        with torch.no_grad():
            valid_mask = bdf_c.detach() <= BDF_MAX
            q = torch.zeros_like(sdf_c.detach())

            if valid_mask.any():
                feat_valid = torch.cat([
                    sdf_c.detach()[valid_mask, None],
                    bdf_c.detach()[valid_mask, None],
                    grad_sdf.detach()[valid_mask],
                    grad_bdf.detach()[valid_mask],
                ], dim=1)
                q[valid_mask] = q_mlp(feat_valid)

            x_pred = apply_bdf_validity_to_xpred(
                sdf_c=sdf_c.detach(),
                bdf_c=bdf_c.detach(),
                q=q,
                bdf_max=BDF_MAX,
            )
            values[start:end] = x_pred.cpu().numpy()

        start = end

    return xs, ys, zs, values.reshape(res, res, res)


# ============================================================
# config
# ============================================================

def _read_json(path: Path):
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if not isinstance(obj, dict):
        raise ValueError(f"Config JSON must be an object/dict, got: {type(obj)}")
    return obj


def _require_vec3(cfg, key: str):
    if key not in cfg:
        raise KeyError(f"Missing required key in config JSON: '{key}'")
    arr = np.asarray(cfg[key], dtype=np.float32)
    if arr.shape != (3,):
        raise ValueError(f"Config key '{key}' must be a 3-element array, got shape={arr.shape}")
    return arr


def _require_path(cfg, key: str):
    if key not in cfg:
        raise KeyError(f"Missing required key in config JSON: '{key}'")
    val = cfg[key]
    if not isinstance(val, str) or len(val.strip()) == 0:
        raise ValueError(f"Config key '{key}' must be a non-empty string path.")
    return Path(val)


def load_runtime_config(config_json_path: Path):
    cfg = _read_json(config_json_path)

    grid_min = _require_vec3(cfg, "grid_min")
    grid_max = _require_vec3(cfg, "grid_max")
    if not np.all(np.isfinite(grid_min)) or not np.all(np.isfinite(grid_max)):
        raise ValueError("grid_min / grid_max must be finite.")
    if not np.all(grid_min < grid_max):
        raise ValueError(f"Require grid_min < grid_max elementwise, got grid_min={grid_min}, grid_max={grid_max}")

    pivot2 = _require_vec3(cfg, "pivot2")
    axis2 = _normalize_axis(_require_vec3(cfg, "axis2"), "axis2")
    pivot3 = _require_vec3(cfg, "pivot3")
    axis3 = _normalize_axis(_require_vec3(cfg, "axis3"), "axis3")

    return {
        "grid_min": grid_min,
        "grid_max": grid_max,
        "pivot2": pivot2,
        "axis2": axis2,
        "pivot3": pivot3,
        "axis3": axis3,
        "ckpt1": _require_path(cfg, "ckpt1"),
        "ckpt2": _require_path(cfg, "ckpt2"),
        "ckpt3": _require_path(cfg, "ckpt3"),
        "q_ckpt": _require_path(cfg, "q_ckpt"),
    }


# ============================================================
# argparser
# ============================================================

def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config-json", type=str, required=True)
    ap.add_argument("--rot2-deg", dest="rot2_deg", type=float, default=0.0)
    ap.add_argument("--rot3-deg", dest="rot3_deg", type=float, default=0.0)
    ap.add_argument("--iso-level", type=float, required=True)
    return ap


# ============================================================
# main
# ============================================================

def main():
    args = build_argparser().parse_args()

    device = torch.device(
        DEFAULT_DEVICE if torch.cuda.is_available() and DEFAULT_DEVICE.startswith("cuda") else "cpu"
    )

    iso_level_abs_max = float(ISO_LEVEL_MAX_FRAC_OF_BDF * BDF_MAX)
    if abs(float(args.iso_level)) > iso_level_abs_max:
        raise ValueError(
            f"Invalid --iso-level={args.iso_level}. "
            f"Require abs(iso-level) <= {ISO_LEVEL_MAX_FRAC_OF_BDF:.2f} * BDF_MAX = {iso_level_abs_max:.6f}."
        )

    runtime_cfg = load_runtime_config(Path(args.config_json))

    grid_min = runtime_cfg["grid_min"]
    grid_max = runtime_cfg["grid_max"]
    pivot2_np = runtime_cfg["pivot2"]
    axis2_np = runtime_cfg["axis2"]
    pivot3_np = runtime_cfg["pivot3"]
    axis3_np = runtime_cfg["axis3"]

    dec1_sdf, dec1_bdf, z1_sdf, z1_bdf = load_model(runtime_cfg["ckpt1"], device)
    dec2_sdf, dec2_bdf, z2_sdf, z2_bdf = load_model(runtime_cfg["ckpt2"], device)
    dec3_sdf, dec3_bdf, z3_sdf, z3_bdf = load_model(runtime_cfg["ckpt3"], device)

    q_mlp = load_q_mlp(
        runtime_cfg["q_ckpt"],
        device,
        fallback_in_dim=Q_FALLBACK_IN_DIM,
        fallback_hidden=Q_FALLBACK_HIDDEN,
        fallback_layers=Q_FALLBACK_LAYERS,
    )

    R2_np = make_R_axis_angle(args.rot2_deg, axis2_np)
    R3_np = make_R_axis_angle(args.rot3_deg, axis3_np)

    pivot2_t = torch.from_numpy(pivot2_np).to(device=device)
    R2_t = torch.from_numpy(R2_np).to(device=device)
    pivot3_t = torch.from_numpy(pivot3_np).to(device=device)
    R3_t = torch.from_numpy(R3_np).to(device=device)

    xs, ys, zs, field_q = evaluate_q_field_on_grid(
        grid_min=grid_min,
        grid_max=grid_max,
        res=GRID_RESOLUTION,
        batch_size=BATCH_SIZE,
        device=device,
        q_mlp=q_mlp,
        dec1_sdf=dec1_sdf, z1_sdf=z1_sdf,
        dec2_sdf=dec2_sdf, z2_sdf=z2_sdf,
        dec3_sdf=dec3_sdf, z3_sdf=z3_sdf,
        dec1_bdf=dec1_bdf, z1_bdf=z1_bdf,
        dec2_bdf=dec2_bdf, z2_bdf=z2_bdf,
        dec3_bdf=dec3_bdf, z3_bdf=z3_bdf,
        pivot2_t=pivot2_t, R2_t=R2_t,
        pivot3_t=pivot3_t, R3_t=R3_t,
    )

    verts, faces = iso_surface_from_field(xs, ys, zs, field_q, level=float(args.iso_level))
    if verts is None:
        raise RuntimeError("Failed to extract iso-surface.")

    out_path = Path.cwd() / "out.obj"
    save_obj(str(out_path), verts, faces)
    print(f"[INFO] Successfully wrote output to: {out_path}")


if __name__ == "__main__":
    main()