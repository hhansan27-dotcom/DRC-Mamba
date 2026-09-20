"""DRC-Mamba network for infrared small target detection.

The implementation directly uses GroupNorm, matching the training configuration.
The implementation follows the paper terminology: response decomposition, candidate preservation, MP-Mamba, and RCD.
"""

from __future__ import annotations

from typing import Dict, Tuple, Optional, Any

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from mamba_ssm import Mamba as _MambaCore
except Exception as exc:  # imported lazily; construction will raise a clear error
    _MambaCore = None
    _MAMBA_IMPORT_ERROR = exc
else:
    _MAMBA_IMPORT_ERROR = None


# ============================================================
# Response-field image operators
# ============================================================

SOBEL_X = torch.tensor(
    [[[-1.0, 0.0, 1.0],
      [-2.0, 0.0, 2.0],
      [-1.0, 0.0, 1.0]]], dtype=torch.float32
).unsqueeze(0)
SOBEL_Y = torch.tensor(
    [[[-1.0, -2.0, -1.0],
      [0.0, 0.0, 0.0],
      [1.0, 2.0, 1.0]]], dtype=torch.float32
).unsqueeze(0)
LAPLACE = torch.tensor(
    [[[0.0, 1.0, 0.0],
      [1.0, -4.0, 1.0],
      [0.0, 1.0, 0.0]]], dtype=torch.float32
).unsqueeze(0)


def _kernel_like(kernel: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return kernel.to(device=x.device, dtype=x.dtype)


def box_blur(x: torch.Tensor, ks: int = 5) -> torch.Tensor:
    pad = ks // 2
    return F.avg_pool2d(x, kernel_size=ks, stride=1, padding=pad)


def sobel_grad(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    gx = F.conv2d(x, _kernel_like(SOBEL_X, x), padding=1)
    gy = F.conv2d(x, _kernel_like(SOBEL_Y, x), padding=1)
    return gx, gy


def laplace(x: torch.Tensor) -> torch.Tensor:
    return F.conv2d(x, _kernel_like(LAPLACE, x), padding=1)


def local_contrast(x: torch.Tensor, ks: int = 7) -> torch.Tensor:
    mu = box_blur(x, ks)
    dev = (x - mu).abs()
    denom = box_blur(dev, ks) + 1e-4
    return dev / denom


def structure_coherence(gx: torch.Tensor, gy: torch.Tensor, smooth_ks: int = 5) -> torch.Tensor:
    jxx = box_blur(gx * gx, smooth_ks)
    jyy = box_blur(gy * gy, smooth_ks)
    jxy = box_blur(gx * gy, smooth_ks)
    delta = torch.sqrt((jxx - jyy).pow(2) + 4.0 * jxy.pow(2) + 1e-6)
    trace = jxx + jyy + 1e-6
    return (delta / trace).clamp(0.0, 1.0)

def log_response(x: torch.Tensor, ks_small: int = 3, ks_large: int = 9) -> torch.Tensor:
    s = box_blur(x, ks_small)
    l = box_blur(x, ks_large)
    y = F.relu(s - l)
    y = y / (box_blur(y, 5) + 1e-4)
    return y.clamp(0.0, 1.0)


def directional_contrast(x: torch.Tensor, ks: int = 9) -> torch.Tensor:
    gx, gy = sobel_grad(x)
    ax, ay = gx.abs(), gy.abs()
    mx, my = box_blur(ax, ks), box_blur(ay, ks)
    y = ((ax - mx).abs() + (ay - my).abs()) / (mx + my + 1e-4)
    return y.clamp(0.0, 1.0)


def pointness_map(x: torch.Tensor) -> torch.Tensor:
    gx, gy = sobel_grad(x)
    coh = structure_coherence(gx, gy)
    lc = local_contrast(x, ks=5)
    logm = log_response(x, ks_small=3, ks_large=9)
    pt = (0.45 * lc + 0.55 * logm) * (1.0 - 0.75 * coh)
    pt = pt / (box_blur(pt, 5) + 1e-4)
    return pt.clamp(0.0, 1.0)
# ============================================================
# Small layers
# ============================================================


class LayerNorm2d(nn.Module):
    def __init__(self, c: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(c))
        self.bias = nn.Parameter(torch.zeros(c))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=1, keepdim=True)
        var = (x - mu).pow(2).mean(dim=1, keepdim=True)
        x = (x - mu) / torch.sqrt(var + self.eps)
        return x * self.weight[:, None, None] + self.bias[:, None, None]


def _group_norm(channels: int, preferred_groups: int = 8) -> nn.GroupNorm:
    """Create GroupNorm using the same group-selection rule used in training."""
    for groups in (preferred_groups, 16, 8, 4, 2, 1):
        if groups <= channels and channels % groups == 0:
            return nn.GroupNorm(groups, channels)
    return nn.GroupNorm(1, channels)


class ConvGNAct(nn.Module):
    """Convolution + GroupNorm + optional SiLU activation."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        k: int = 3,
        s: int = 1,
        p: Optional[int] = None,
        groups: int = 1,
        act: bool = True,
    ):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(in_c, out_c, k, s, p, bias=False, groups=groups)
        self.norm = _group_norm(out_c)
        self.act = nn.SiLU(inplace=True) if act else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(self.norm(self.conv(x)))


class ResidualBlock(nn.Module):
    def __init__(self, c: int, hidden_mult: float = 1.0, dilation: int = 1):
        super().__init__()
        hidden = max(c, int(c * hidden_mult))
        self.conv1 = ConvGNAct(c, hidden, 3, 1, dilation, act=True)
        self.dw = ConvGNAct(hidden, hidden, 3, 1, dilation, groups=hidden, act=True)
        self.conv2 = ConvGNAct(hidden, c, 1, 1, 0, act=False)
        self.act = nn.SiLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.act(x + self.conv2(self.dw(self.conv1(x))))


class Downsample(nn.Module):
    def __init__(self, in_c: int, out_c: int):
        super().__init__()
        self.op = nn.Sequential(
            ConvGNAct(in_c, out_c, 3, 2),
            ResidualBlock(out_c),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.op(x)


class UpFuseBlock(nn.Module):
    def __init__(self, in_c: int, skip_c: int, out_c: int, field_c: int):
        super().__init__()
        self.fuse = nn.Sequential(
            ConvGNAct(in_c + skip_c + field_c, out_c, 3, 1),
            ResidualBlock(out_c),
            ResidualBlock(out_c, hidden_mult=1.2),
        )

    def forward(self, x: torch.Tensor, skip: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        field = F.interpolate(field, size=skip.shape[-2:], mode='bilinear', align_corners=False)
        return self.fuse(torch.cat([x, skip, field], dim=1))


# ============================================================
# Lightweight sequence mixers / path orders
# ============================================================


class SharedBidirectionalMamba(nn.Module):
    """Bidirectional Mamba mixer used by each spatial scan path."""

    def __init__(self, d_model: int, d_state: int = 16, expand: int = 2):
        super().__init__()
        if _MambaCore is None:
            raise ImportError(
                "mamba-ssm is required to build DRC-Mamba. Install the dependencies "
                "from requirements.txt before constructing the model."
            ) from _MAMBA_IMPORT_ERROR
        self.fwd = _MambaCore(d_model=d_model, d_state=d_state, expand=expand)
        self.bwd = _MambaCore(d_model=d_model, d_state=d_state, expand=expand)
        self.out = nn.Linear(d_model * 2, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y_fwd = self.fwd(x)
        y_bwd = torch.flip(self.bwd(torch.flip(x, dims=[1])), dims=[1])
        return self.out(torch.cat([y_fwd, y_bwd], dim=-1))

def _snake_indices(h: int, w: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor]:
    rows = []
    for i in range(h):
        row = torch.arange(i * w, (i + 1) * w, device=device)
        if i % 2 == 1:
            row = torch.flip(row, dims=[0])
        rows.append(row)
    idx = torch.cat(rows, dim=0)
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), device=device)
    return idx, inv


def _diag_indices(h: int, w: int, device: torch.device, anti: bool = False) -> Tuple[torch.Tensor, torch.Tensor]:
    coords = []
    for s in range(h + w - 1):
        cur = []
        for i in range(h):
            j = s - i
            if 0 <= j < w:
                jj = (w - 1 - j) if anti else j
                cur.append(i * w + jj)
        if s % 2 == 1:
            cur = cur[::-1]
        coords.extend(cur)
    idx = torch.tensor(coords, dtype=torch.long, device=device)
    inv = torch.empty_like(idx)
    inv[idx] = torch.arange(idx.numel(), device=device)
    return idx, inv


class MPMambaBlock(nn.Module):
    def __init__(self, c: int, d_state: int = 16, expand: int = 2,
                 orders: Tuple[str, ...] = ('row', 'col', 'diag', 'anti', 'snake')):
        super().__init__()
        self.orders = tuple(orders)
        self.local = nn.Sequential(
            LayerNorm2d(c),
            ConvGNAct(c, c, 3, 1, groups=c),
            ConvGNAct(c, c, 1, 1, 0),
        )
        self.mixer = SharedBidirectionalMamba(d_model=c, d_state=d_state, expand=expand)
        self.out = nn.Sequential(
            nn.Conv2d(c * 2, c, 1, bias=False),
            _group_norm(c),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, c, 1, bias=False),
        )

    @staticmethod
    def _to_sequence(x: torch.Tensor, order: str) -> Tuple[torch.Tensor, Any]:
        b, c, h, w = x.shape
        if order == 'row':
            seq = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            def restore(t: torch.Tensor) -> torch.Tensor:
                return t.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            return seq, restore
        if order == 'col':
            seq = x.permute(0, 3, 2, 1).reshape(b, w * h, c)
            def restore(t: torch.Tensor) -> torch.Tensor:
                return t.view(b, w, h, c).permute(0, 3, 2, 1).contiguous()
            return seq, restore
        if order == 'snake':
            idx, inv = _snake_indices(h, w, x.device)
            flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            seq = flat.index_select(1, idx)
            def restore(t: torch.Tensor) -> torch.Tensor:
                flat_row = t.index_select(1, inv)
                return flat_row.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            return seq, restore
        if order == 'diag':
            idx, inv = _diag_indices(h, w, x.device, anti=False)
            flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            seq = flat.index_select(1, idx)
            def restore(t: torch.Tensor) -> torch.Tensor:
                flat_row = t.index_select(1, inv)
                return flat_row.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            return seq, restore
        if order == 'anti':
            idx, inv = _diag_indices(h, w, x.device, anti=True)
            flat = x.permute(0, 2, 3, 1).reshape(b, h * w, c)
            seq = flat.index_select(1, idx)
            def restore(t: torch.Tensor) -> torch.Tensor:
                flat_row = t.index_select(1, inv)
                return flat_row.view(b, h, w, c).permute(0, 3, 1, 2).contiguous()
            return seq, restore
        raise ValueError(f'Unsupported order: {order}')

    def forward(
        self, x: torch.Tensor, memory: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        local = self.local(x)
        outs, mems = [], []
        for order in self.orders:
            seq, restore = self._to_sequence(x, order)
            if memory is not None:
                seq_in = torch.cat([memory, seq], dim=1)
                mixed = self.mixer(seq_in)
                mems.append(mixed[:, :memory.shape[1], :])
                s = mixed[:, memory.shape[1]:, :]
            else:
                s = self.mixer(seq)
            outs.append(restore(s))
        global_feat = torch.stack(outs, dim=0).mean(dim=0)
        y = self.out(torch.cat([local, global_feat], dim=1))
        new_mem = None
        if len(mems) > 0:
            new_mem = torch.stack(mems, dim=0).mean(dim=0)
        return y, new_mem


class SceneTokenReader(nn.Module):
    def __init__(self, c: int):
        super().__init__()
        self.q = nn.Conv2d(c, c, 1, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.proj = nn.Conv2d(c, c, 1, bias=False)
        self.scale = c ** -0.5

    def forward(self, x: torch.Tensor, tokens: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        q = self.q(x).flatten(2).transpose(1, 2)
        k = self.k(tokens)
        v = self.v(tokens)
        attn = torch.softmax(torch.matmul(q, k.transpose(1, 2)) * self.scale, dim=-1)
        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).reshape(b, c, h, w)
        return self.proj(out)


class SceneTokenBuilder(nn.Module):
    def __init__(self, c: int, num_tokens: int):
        super().__init__()
        self.num_tokens = int(num_tokens)
        self.refine = nn.Sequential(
            nn.Linear(c, c),
            nn.GELU(),
            nn.Linear(c, c),
        )

    def forward(self, feat: torch.Tensor, score: torch.Tensor) -> torch.Tensor:
        score = score.clamp_min(1e-4)
        tok = F.adaptive_avg_pool2d(feat * score, (self.num_tokens, 1)).squeeze(-1).permute(0, 2, 1)
        den = F.adaptive_avg_pool2d(score, (self.num_tokens, 1)).squeeze(-1).permute(0, 2, 1)
        tok = tok / den.clamp_min(1e-4)
        return self.refine(tok)

class RepeatBackgroundGate(nn.Module):
    def __init__(self, c: int, num_tokens: int = 6):
        super().__init__()
        self.token_builder = SceneTokenBuilder(c, num_tokens)
        self.reader = SceneTokenReader(c)
        self.score = nn.Sequential(
            ConvGNAct(c * 2, c, 3, 1),
            nn.Conv2d(c, 1, 1)
        )

    def forward(self, feat: torch.Tensor, repeat_hint: torch.Tensor):
        tokens = self.token_builder(feat, repeat_hint)
        bg_read = self.reader(feat, tokens)
        score = torch.sigmoid(self.score(torch.cat([feat, bg_read], dim=1)))
        return score, tokens

class BackgroundReconstructionModule(nn.Module):
    """
    Multi-scale background reconstruction and veil estimation used by response decomposition.
    """
    def __init__(self, mid_c: int = 16):
        super().__init__()
        self.encoder = nn.Sequential(
            ConvGNAct(6, mid_c, 3, 1),
            ResidualBlock(mid_c),
            ResidualBlock(mid_c, hidden_mult=1.2),
        )
        self.mix = nn.Conv2d(mid_c, 3, 1, bias=True)
        self.bg_res = nn.Conv2d(mid_c, 1, 1, bias=True)
        self.veil_head = nn.Conv2d(mid_c, 1, 1, bias=True)
        self.struct_head = nn.Conv2d(mid_c, 1, 1, bias=True)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        b5 = box_blur(x, 5)
        b9 = box_blur(x, 9)
        b15 = box_blur(x, 15)

        gx, gy = sobel_grad(x)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        lap = laplace(x).abs()

        feat = torch.cat([x, b5, b9, b15, grad, lap], dim=1)
        z = self.encoder(feat)

        # Background seed from multi-scale smoothing with a small learnable correction.
        w = torch.softmax(self.mix(z), dim=1)
        bg_seed = w[:, 0:1] * b5 + w[:, 1:2] * b9 + w[:, 2:3] * b15

        bg = (bg_seed + 0.15 * torch.tanh(self.bg_res(z))).clamp(0.0, 1.0)

        # Slowly varying low-frequency veil.
        veil_seed = (b15 - b5).abs()
        veil = (torch.sigmoid(self.veil_head(z)) * veil_seed).clamp(0.0, 1.0)

        clear = torch.clamp(x - 0.35 * veil, 0.0, 1.0)
        recon_err = (x - bg).abs().clamp(0.0, 1.0)
        sparse = F.relu(clear - bg)

        bg_struct = (
            0.50 * structure_coherence(*sobel_grad(bg)) +
            0.25 * directional_contrast(bg, ks=7) +
            0.25 * torch.sigmoid(self.struct_head(z))
        ).clamp(0.0, 1.0)

        return {
            "bg": bg,                 # B_hat
            "veil": veil,             # V_hat
            "clear": clear,
            "sparse": sparse,         # sparse residual approximation
            "recon_err": recon_err,   # |x - B_hat|
            "bg_struct": bg_struct,
        }

# ============================================================
# Response decomposition
# ============================================================


class ResponseDecompositionModule(nn.Module):
    """
    Construct three complementary response fields:
    - Fs: compact-source field
    - Fb: background-structure field
    - Ff: interference-sensitive field

    These are overlapping response descriptions, not foreground probabilities.
    """
    def __init__(self):
        super().__init__()
        self.background_reconstruction = BackgroundReconstructionModule()

    def _packs(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        reconstruction = self.background_reconstruction(x)
        x_clear = reconstruction["clear"]
        bg = reconstruction["bg"]
        veil = reconstruction["veil"]
        sparse = reconstruction["sparse"]
        recon_err = reconstruction["recon_err"]
        bg_struct = reconstruction["bg_struct"]

        gx, gy = sobel_grad(x_clear)
        grad = torch.sqrt(gx * gx + gy * gy + 1e-6)
        lap = laplace(x_clear).abs()
        coh = structure_coherence(gx, gy)
        contrast = local_contrast(x_clear, ks=7)
        dirc = directional_contrast(x_clear, ks=9)
        point = pointness_map(x_clear)
        logm = log_response(x_clear, ks_small=3, ks_large=9)
        blob = (torch.exp(-lap) * logm).clamp(0.0, 1.0)

        # Estimate line structure from reconstructed background so that target details
        # are not mistaken for structural support.
        bg_gx, bg_gy = sobel_grad(bg)
        bg_grad = torch.sqrt(bg_gx * bg_gx + bg_gy * bg_gy + 1e-6)
        bg_coh = structure_coherence(bg_gx, bg_gy)
        line = (bg_coh * (bg_grad / (box_blur(bg_grad, 5) + 1e-4))).clamp(0.0, 1.0)

        # Compact-source response field.
        fs = torch.cat([
            sparse,          # residual-like
            contrast,
            point,
            blob,
            torch.exp(-lap)
        ], dim=1)

        # Background-structure response field.
        fb = torch.cat([
            bg,
            bg_struct,
            bg_coh,
            line,
            veil
        ], dim=1)

        # Interference-sensitive response field.
        ff = torch.cat([
            recon_err,
            line,
            bg_coh,
            dirc,
            (0.5 * grad + 0.5 * lap).clamp(0.0, 1.0)
        ], dim=1)

        # Keep the base representation at 10 channels for the encoder stem.
        base = torch.cat([
            x_clear,
            sparse,
            bg,
            recon_err,
            point,
            contrast,
            bg_struct,
            line,
            bg_coh,
            veil
        ], dim=1)

        return {
            'fs': fs,
            'fb': fb,
            'ff': ff,
            'base': base,
            'clear': x_clear,
            'bg': bg,
            'veil': veil,
            'sparse': sparse,
            'recon_err': recon_err,
            'bg_struct': bg_struct,
        }

    def forward(self, x: torch.Tensor) -> Dict[str, Dict[str, torch.Tensor]]:
        full = self._packs(x)
        out = {'full': full}
        scale_map = {
            'half': 0.5,
            'quarter': 0.25,
            'eighth': 0.125,
            'sixteenth': 0.0625,
            'thirty_second': 0.03125,
        }
        for name, scale in scale_map.items():
            out[name] = {
                'fs': F.interpolate(full['fs'], scale_factor=scale, mode='bilinear', align_corners=False),
                'fb': F.interpolate(full['fb'], scale_factor=scale, mode='bilinear', align_corners=False),
                'ff': F.interpolate(full['ff'], scale_factor=scale, mode='bilinear', align_corners=False),
                'base': F.interpolate(full['base'], scale_factor=scale, mode='bilinear', align_corners=False),
            }
        return out


class CandidateDetailPath(nn.Module):
    def __init__(self, fs_c: int = 5, fb_c: int = 5, dims: Tuple[int, int, int] = (24, 32, 48)):
        super().__init__()
        d0, d1, d2 = dims
        self.fs_full = nn.Sequential(ConvGNAct(1 + fs_c, d0, 3, 1), ResidualBlock(d0))
        self.fb_full = nn.Sequential(ConvGNAct(1 + fb_c, d0, 3, 1), ResidualBlock(d0))
        self.fs_down1 = nn.Sequential(Downsample(d0, d1), ResidualBlock(d1))
        self.fb_down1 = nn.Sequential(Downsample(d0, d1), ResidualBlock(d1))
        self.fs_down2 = nn.Sequential(Downsample(d1, d2), ResidualBlock(d2))
        self.fb_down2 = nn.Sequential(Downsample(d1, d2), ResidualBlock(d2))
        self.full_mix = nn.Sequential(ConvGNAct(d0 * 2, d0, 1, 1, 0), ResidualBlock(d0))
        self.half_mix = nn.Sequential(ConvGNAct(d1 * 2, d1, 1, 1, 0), ResidualBlock(d1))
        self.quarter_mix = nn.Sequential(ConvGNAct(d2 * 2, d2, 1, 1, 0), ResidualBlock(d2))

    def forward(self, x: torch.Tensor, phys_full: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        s_full = self.fs_full(torch.cat([x, phys_full['fs']], dim=1))
        c_full = self.fb_full(torch.cat([x, phys_full['fb']], dim=1))
        s_half = self.fs_down1(s_full)
        c_half = self.fb_down1(c_full)
        s_quarter = self.fs_down2(s_half)
        c_quarter = self.fb_down2(c_half)
        return {
            'fs_full': s_full,
            'fb_full': c_full,
            'fs_half': s_half,
            'fb_half': c_half,
            'fs_quarter': s_quarter,
            'fb_quarter': c_quarter,
            'full': self.full_mix(torch.cat([s_full, c_full], dim=1)),
            'half': self.half_mix(torch.cat([s_half, c_half], dim=1)),
            'quarter': self.quarter_mix(torch.cat([s_quarter, c_quarter], dim=1)),
        }

class CandidateRetentionPath(nn.Module):
    """
    Candidate path of the candidate-preservation branch.
    It forms a learned retention response and a structure-aware preservation floor.
    """
    def __init__(self, fs_channels: int = 5, ff_channels: int = 5, hid_c: int = 24):
        super().__init__()
        self.body = nn.Sequential(
            ConvGNAct(1 + 1 + fs_channels + ff_channels, hid_c, 3, 1),
            ResidualBlock(hid_c),
            ResidualBlock(hid_c, hidden_mult=1.2),
        )
        self.point_head = nn.Conv2d(hid_c, 1, 1)
        self.center_head = nn.Conv2d(hid_c, 1, 1)
        self.tiny_head = nn.Conv2d(hid_c, 1, 1)

        self.retention_head = nn.Conv2d(hid_c, 1, 1)
        self.retention_gate_head = nn.Conv2d(hid_c, 1, 1)

    def forward(
        self,
        x: torch.Tensor,
        fs_field: torch.Tensor,
        ff_field: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        lc = local_contrast(x, ks=5)
        z = self.body(torch.cat([x, lc, fs_field, ff_field], dim=1))

        point_prior = torch.sigmoid(self.point_head(z))
        center_prior = torch.sigmoid(self.center_head(z + 0.25 * z * point_prior))
        tiny_target_map = torch.sigmoid(
            self.tiny_head(z + 0.20 * z * center_prior + 0.10 * z * point_prior)
        )

        structure_response = (
            0.35 * ff_field[:, 1:2] +
            0.25 * ff_field[:, 2:3] +
            0.20 * ff_field[:, 0:1]
        ).clamp(0.0, 1.0)

        pointness = fs_field[:, 2:3].clamp(0.0, 1.0)
        sparse_hint = fs_field[:, 0:1].clamp(0.0, 1.0)

        candidate_score = (
            0.30 * point_prior +
            0.28 * center_prior +
            0.26 * tiny_target_map +
            0.16 * pointness
        ).clamp(0.0, 1.0)

        retention_response = torch.sigmoid(
            self.retention_head(
                z + 0.30 * candidate_score + 0.15 * sparse_hint - 0.12 * structure_response
            )
        )
        retention_gate = torch.sigmoid(
            self.retention_gate_head(
                z + 0.25 * retention_response + 0.20 * center_prior + 0.15 * tiny_target_map - 0.10 * structure_response
            )
        )

        preservation_floor = candidate_score * torch.sigmoid(4.0 * (0.58 - structure_response))
        retention_response = torch.maximum(retention_response, 0.65 * preservation_floor)
        retention_gate = torch.maximum(
            retention_gate,
            (0.55 * retention_response + 0.25 * center_prior + 0.20 * tiny_target_map).clamp(0.0, 1.0)
        )

        point_prior = (point_prior * (1.0 - 0.10 * structure_response)).clamp(0.0, 1.0)
        center_prior = (center_prior * (1.0 - 0.08 * structure_response)).clamp(0.0, 1.0)
        tiny_target_map = (tiny_target_map * (1.0 - 0.06 * structure_response)).clamp(0.0, 1.0)

        return {
            "point_prior": point_prior,
            "center_prior": center_prior,
            "tiny_target_map": tiny_target_map,
            "retention_response": retention_response,
            "retention_gate": retention_gate,
        }


class CandidatePreservationBranch(nn.Module):
    """Paper-level CP branch containing the detail path and candidate path."""

    def __init__(
        self,
        fs_channels: int = 5,
        fb_channels: int = 5,
        ff_channels: int = 5,
        detail_dims: Tuple[int, int, int] = (24, 32, 48),
        retention_hidden: int = 24,
    ):
        super().__init__()
        self.detail_path = CandidateDetailPath(
            fs_c=fs_channels, fb_c=fb_channels, dims=detail_dims
        )
        self.candidate_path = CandidateRetentionPath(
            fs_channels=fs_channels, ff_channels=ff_channels, hid_c=retention_hidden
        )

    def forward(
        self, x: torch.Tensor, response_fields: Dict[str, Dict[str, torch.Tensor]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        detail = self.detail_path(x, response_fields["full"])
        retention = self.candidate_path(
            x, response_fields["full"]["fs"], response_fields["full"]["ff"]
        )
        return detail, retention

# ============================================================
# Stage 1: response-conditioned local encoding
# ============================================================


class EncoderStage1(nn.Module):
    def __init__(
        self,
        c: int,
        detail_c: int,
        fs_channels: int = 5,
        fb_channels: int = 5,
        ff_channels: int = 5
    ):
        super().__init__()
        self.fs_detail_proj = nn.Conv2d(detail_c, c, 1, bias=False)
        self.fb_detail_proj = nn.Conv2d(detail_c, c, 1, bias=False)
        self.fs_field_proj = nn.Conv2d(fs_channels, c, 1, bias=False)
        self.fb_field_proj = nn.Conv2d(fb_channels, c, 1, bias=False)
        self.ff_field_proj = nn.Conv2d(ff_channels, c, 1, bias=False)

        self.prior_proj = nn.Conv2d(4, c, 1, bias=False)
        self.retention_logit = nn.Conv2d(c, 1, 1)

        self.body = nn.Sequential(
            ConvGNAct(c * 7, c, 3, 1),
            ResidualBlock(c),
            ResidualBlock(c, hidden_mult=1.2),
        )

        self.source_logit = nn.Conv2d(c, 1, 1)
        self.source_confidence_logit = nn.Conv2d(c, 1, 1)
        self.nuis_logit = nn.Conv2d(c, 1, 1)
        self.clutter_seed_logit = nn.Conv2d(c, 1, 1)
        self.line_reject_logit = nn.Conv2d(c, 1, 1)
        self.interference_seed_logit = nn.Conv2d(c, 1, 1)
        self.center_seed_logit = nn.Conv2d(c, 1, 1)
        self.bnd_logits = nn.Conv2d(c, 1, 1)
        self.det_head = nn.Conv2d(c, 1, 1)
        self.q0_head = nn.Conv2d(c, 1, 1)

        self.mix = nn.Sequential(
            nn.Conv2d(4, c, 1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, 1)
        )
        self.progress = 0.0

    def set_progress(self, epoch: int, max_epoch: int):
        self.progress = min(1.0, float(epoch) / float(max(1, max_epoch)))

    def forward(
        self,
        x: torch.Tensor,
        detail_fs: torch.Tensor,
        detail_fb: torch.Tensor,
        fs_field: torch.Tensor,
        fb_field: torch.Tensor,
        ff_field: torch.Tensor,
        point_prior: torch.Tensor,
        center_prior: torch.Tensor,
        tiny_target_prior: torch.Tensor,
        retention_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ramp = 0.15 + 0.85 * self.progress

        ds = self.fs_detail_proj(detail_fs)
        dc = self.fb_detail_proj(detail_fb)
        ps = self.fs_field_proj(fs_field)
        pc = self.fb_field_proj(fb_field)
        pf = self.ff_field_proj(ff_field)

        rp = self.prior_proj(torch.cat(
            [point_prior, center_prior, tiny_target_prior, retention_prior], dim=1
        ))

        source_bias = ds + ps + 0.34 * rp
        clutter_bias = dc + pc + 0.18 * pf

        z = self.body(torch.cat([x, ds, dc, ps, pc, pf, rp], dim=1))
        z = z + (0.18 + 0.20 * ramp) * source_bias - (0.05 + 0.08 * ramp) * clutter_bias

        source_logit = self.source_logit(z + 0.22 * source_bias)
        source_confidence_logit = self.source_confidence_logit(z)
        nuis_logit = self.nuis_logit(z + 0.08 * clutter_bias)
        clutter_seed_logit = self.clutter_seed_logit(z + 0.16 * clutter_bias - 0.08 * source_bias)
        line_reject_logit = self.line_reject_logit(z + 0.18 * clutter_bias)
        interference_seed_logit = self.interference_seed_logit(z + 0.24 * pf + 0.08 * clutter_bias)
        center_seed_logit = self.center_seed_logit(z + 0.22 * rp + 0.10 * source_bias)
        bnd_logits = self.bnd_logits(z + 0.18 * source_bias)
        retention_seed_logit = self.retention_logit(z + 0.28 * rp + 0.10 * source_bias - 0.05 * clutter_bias)

        det_raw = torch.sigmoid(self.det_head(z))
        source_probability = torch.sigmoid(source_logit)
        nuis_prob = torch.sigmoid(nuis_logit)
        clutter_seed = torch.sigmoid(clutter_seed_logit)
        line_reject = torch.sigmoid(line_reject_logit)
        interference_seed = torch.sigmoid(interference_seed_logit)
        center_seed = torch.sigmoid(center_seed_logit)
        source_confidence = torch.sigmoid(source_confidence_logit)
        bnd = torch.sigmoid(bnd_logits)
        retention_seed = torch.sigmoid(retention_seed_logit)

        cand_logits = self.mix(torch.cat([
            source_logit,
            -0.70 * nuis_logit,
            -0.50 * line_reject_logit,
            0.35 * source_confidence_logit
        ], dim=1))
        cand_map = torch.sigmoid(cand_logits)
        q0 = torch.sigmoid(self.q0_head(z))

        support_seed = (
            0.24 * source_probability +
            0.16 * cand_map +
            0.08 * bnd +
            0.12 * point_prior +
            0.12 * center_seed +
            0.14 * tiny_target_prior +
            0.14 * retention_seed
        ).clamp(0.0, 1.0)

        contradict = (
            (source_probability - center_seed).abs() +
            0.5 * (source_confidence - center_seed).abs()
        ).clamp(0.0, 1.0)

        risk_seed = (
            0.24 * nuis_prob +
            0.18 * clutter_seed +
            0.16 * line_reject +
            0.18 * interference_seed +
            0.12 * contradict +
            0.12 * (1.0 - tiny_target_prior)
        ).clamp(0.0, 1.0)

        g_det_base = (
            det_raw *
            (0.88 + 0.12 * source_confidence) *
            (1.0 - 0.10 * clutter_seed) *
            (1.0 - 0.04 * line_reject) *
            (1.0 - 0.05 * interference_seed)
        ).clamp(0.0, 1.0)

        preservation_floor = (
            (0.42 * retention_prior + 0.24 * retention_seed + 0.18 * tiny_target_prior + 0.16 * center_seed) *
            (1.0 - 0.18 * line_reject) *
            (1.0 - 0.20 * interference_seed)
        ).clamp(0.0, 1.0)

        g_det = torch.maximum(g_det_base, preservation_floor)

        coh = fb_field[:, 2:3].clamp(0.0, 1.0)
        line = fb_field[:, 3:4].clamp(0.0, 1.0)

        clutter = torch.clamp(
            0.34 * clutter_seed +
            0.18 * nuis_prob +
            0.16 * line +
            0.14 * interference_seed +
            0.18 * contradict,
            0.0, 1.0
        )

        rejection = torch.clamp(
            0.28 * risk_seed +
            0.16 * line +
            0.16 * interference_seed +
            0.12 * (1.0 - source_confidence) +
            0.12 * (1.0 - center_seed) +
            0.16 * contradict,
            0.0, 1.0
        )

        support = torch.maximum(support_seed, 0.80 * preservation_floor).clamp(0.0, 1.0)
        uncertainty = 1.0 - source_confidence

        pack = {
            "source_logit": source_logit,
            "source_confidence_logit": source_confidence_logit,
            "source_confidence": source_confidence,
            "nuis_logit": nuis_logit,
            "clutter_seed_logit": clutter_seed_logit,
            "line_reject_logit": line_reject_logit,
            "interference_seed_logit": interference_seed_logit,
            "center_seed_logit": center_seed_logit,
            "cand_logits": cand_logits,
            "cand_map": cand_map,
            "bnd_logits": bnd_logits,
            "bnd": bnd,
            "g_det": g_det,
            "g": g_det,
            "q0": q0,
            "source_probability": source_probability,
            "clutter_seed": clutter_seed,
            "line_reject": line_reject,
            "interference_seed": interference_seed,
            "center_seed": center_seed,
            "point_prior": point_prior,
            "tiny_target_prior": tiny_target_prior,
            "retention_prior": retention_prior,
            "retention_seed": retention_seed,
            "preservation_floor": preservation_floor,
            "support_seed": support_seed,
            "risk_seed": risk_seed,
            "contradict": contradict,
            "e_pos": source_probability * source_confidence,
            "e_neg": 0.30 * nuis_prob + 0.22 * clutter_seed + 0.18 * line_reject + 0.16 * interference_seed + 0.14 * contradict,
            "u": uncertainty,

            "support": support,
            "clutter": clutter,
            "rejection": rejection,
            "uncertainty": uncertainty,
            "line": line,
            "coh": coh,
        }
        return z, pack


class EncoderStage2(nn.Module):
    """Stage 2 encoder with response-conditioned scene memory and MP-Mamba."""
    def __init__(
        self,
        c: int,
        fs_channels: int = 5,
        fb_channels: int = 5,
        background_tokens: int = 6,
        structure_tokens: int = 6,
        source_tokens: int = 4,
        d_state: int = 16,
        expand: int = 2
    ):
        super().__init__()
        self.fs_field_proj = nn.Conv2d(fs_channels, c, 1, bias=False)
        self.fb_field_proj = nn.Conv2d(fb_channels, c, 1, bias=False)
        self.prior_proj = nn.Conv2d(2, c, 1, bias=False)

        self.background_token_builder = SceneTokenBuilder(c, background_tokens)
        self.structure_token_builder = SceneTokenBuilder(c, structure_tokens)
        self.source_token_builder = SceneTokenBuilder(c, source_tokens)

        self.mp_mamba = MPMambaBlock(c, d_state=d_state, expand=expand)
        self.background_token_reader = SceneTokenReader(c)
        self.structure_token_reader = SceneTokenReader(c)
        self.source_token_reader = SceneTokenReader(c)

        self.fuse = nn.Sequential(
            ConvGNAct(c * 5, c, 3, 1),
            ResidualBlock(c)
        )

        # connected-background gate
        self.conn_bg_head = nn.Sequential(
            ConvGNAct(c + 3, c, 3, 1),
            nn.Conv2d(c, 1, 1)
        )

        self.clutter_logit = nn.Conv2d(c, 1, 1)
        self.rejection_logit = nn.Conv2d(c, 1, 1)
        self.ref_conf = nn.Conv2d(c, 1, 1)
        self.det_head = nn.Conv2d(c, 1, 1)
        self.pos_head = nn.Conv2d(c, 1, 1)
        self.neg_head = nn.Conv2d(c, 1, 1)
        self.unc_head = nn.Conv2d(c, 1, 1)

        # Sparse hardest-negative discrimination.
        self.hard_neg_head = nn.Sequential(
            ConvGNAct(c + 4, c, 3, 1),
            nn.Conv2d(c, 1, 1)
        )

        # Propagate the retained-candidate response to later stages and the decoder.
        self.retention_gate_head = nn.Conv2d(c, 1, 1)

        self.progress = 0.0

    def set_progress(self, epoch: int, max_epoch: int):
        self.progress = min(1.0, float(epoch) / float(max(1, max_epoch)))

    def forward(
        self,
        x: torch.Tensor,
        fs_field: torch.Tensor,
        fb_field: torch.Tensor,
        source_prior: torch.Tensor,
        clutter_hint: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ramp = 0.15 + 0.85 * self.progress

        ps = self.fs_field_proj(fs_field)
        pc = self.fb_field_proj(fb_field)
        prior = self.prior_proj(torch.cat([source_prior, clutter_hint], dim=1))

        x0 = x + (0.10 + 0.15 * ramp) * ps - (0.05 + 0.08 * ramp) * pc + prior

        score_background = (1.0 - source_prior).clamp(0.0, 1.0)
        score_structure = clutter_hint.clamp(0.0, 1.0)
        score_source = (source_prior * (1.0 - clutter_hint)).clamp(0.0, 1.0)

        background_tokens = self.background_token_builder(x0, score_background)
        structure_tokens = self.structure_token_builder(x0, score_structure)
        source_tokens = self.source_token_builder(x0, score_source)

        memory = torch.cat([background_tokens, structure_tokens, source_tokens], dim=1)
        y, memory = self.mp_mamba(x0, memory)

        n_background, n_structure, n_source = background_tokens.shape[1], structure_tokens.shape[1], source_tokens.shape[1]
        background_tokens = memory[:, :n_background]
        structure_tokens = memory[:, n_background:n_background + n_structure]
        source_tokens = memory[:, n_background + n_structure:n_background + n_structure + n_source]

        background_read = self.background_token_reader(x + y, background_tokens)
        structure_read = self.structure_token_reader(x + y, structure_tokens)
        source_read = self.source_token_reader(x + y, source_tokens)

        z = self.fuse(torch.cat([x + y, background_read, structure_read, source_read, ps - pc], dim=1))

        coh = fb_field[:, 2:3].clamp(0.0, 1.0)
        line = fb_field[:, 3:4].clamp(0.0, 1.0)
        veil = fb_field[:, 4:5].clamp(0.0, 1.0)

        conn_in = torch.cat([
            z,
            F.interpolate(coh, size=z.shape[-2:], mode="bilinear", align_corners=False),
            F.interpolate(line, size=z.shape[-2:], mode="bilinear", align_corners=False),
            F.interpolate(veil, size=z.shape[-2:], mode="bilinear", align_corners=False),
        ], dim=1)
        conn_bg = torch.sigmoid(self.conn_bg_head(conn_in))

        clutter_prob = torch.sigmoid(
            self.clutter_logit(z + structure_read - 0.5 * source_read + 0.25 * conn_bg)
        )
        rejection_prob = torch.sigmoid(
            self.rejection_logit(z + 0.50 * structure_read - 0.25 * source_read + 0.25 * conn_bg)
        )
        ref_conf = torch.sigmoid(self.ref_conf(z))

        contradict = (source_prior - ref_conf).abs().clamp(0.0, 1.0)

        hard_neg = torch.sigmoid(
            self.hard_neg_head(torch.cat([z, conn_bg, clutter_prob, contradict, line], dim=1))
        )
        hard_neg = (
            hard_neg *
            torch.sigmoid(
                6.0 * (
                    hard_neg -
                    F.avg_pool2d(hard_neg, kernel_size=5, stride=1, padding=2)
                )
            )
        ).clamp(0.0, 1.0)

        retained_response64 = torch.sigmoid(self.retention_gate_head(z + 0.25 * source_prior - 0.10 * hard_neg))

        e_pos = torch.sigmoid(self.pos_head(z + source_read - structure_read)) * (
            0.58 * source_prior + 0.18 * (1.0 - clutter_prob) + 0.24 * ref_conf
        )
        e_neg = torch.sigmoid(self.neg_head(z + structure_read - source_read)) * (
            0.28 + 0.32 * clutter_prob + 0.18 * conn_bg + 0.22 * hard_neg
        )
        unc = torch.sigmoid(self.unc_head((source_read - background_read).abs() + (structure_read - background_read).abs()))

        g_det_base = (
            torch.sigmoid(self.det_head(z)) *
            (1.0 - (0.12 + 0.26 * ramp) * clutter_prob) *
            (1.0 - 0.10 * conn_bg) *
            (1.0 - 0.14 * hard_neg) *
            (0.74 + 0.26 * ref_conf)
        ).clamp(0.0, 1.0)

        support_floor = (
            (0.62 * source_prior + 0.38 * retained_response64) *
            (1.0 - 0.25 * hard_neg) *
            (0.70 + 0.30 * ref_conf)
        ).clamp(0.0, 1.0)

        g_det = torch.maximum(g_det_base, 0.40 * support_floor)

        far_reject = (
            0.30 * clutter_prob +
            0.20 * conn_bg +
            0.30 * hard_neg +
            0.20 * (1.0 - source_prior)
        ).clamp(0.0, 1.0)

        support_hint = (
            e_pos *
            (1.0 - 0.25 * clutter_prob) *
            (1.0 - 0.18 * conn_bg) *
            (1.0 - 0.20 * hard_neg) *
            (0.65 + 0.35 * ref_conf)
        ).clamp(0.0, 1.0)
        support_hint = torch.maximum(support_hint, 0.50 * support_floor)

        clutter = (
            0.28 * clutter_prob +
            0.18 * conn_bg +
            0.28 * hard_neg +
            0.10 * line +
            0.08 * contradict +
            0.08 * unc
        ).clamp(0.0, 1.0)

        rejection = (
            0.26 * rejection_prob +
            0.18 * conn_bg +
            0.30 * hard_neg +
            0.10 * contradict +
            0.08 * unc +
            0.08 * line
        ).clamp(0.0, 1.0)

        pack = {
            "clutter_logit": self.clutter_logit(z + structure_read - 0.5 * source_read),
            "clutter_prob": clutter_prob,
            "rejection_prob": rejection_prob,
            "ref_conf": ref_conf,
            "background_read": background_read,
            "structure_read": structure_read,
            "source_read": source_read,
            "scene_read": (background_read + structure_read + source_read) / 3.0,
            "scene_tokens": memory,
            "background_tokens": background_tokens,
            "structure_tokens": structure_tokens,
            "source_tokens": source_tokens,
            "g_det": g_det,
            "g": g_det,
            "e_pos": e_pos,
            "e_neg": e_neg,
            "u": unc,
            "far_reject": far_reject,
            "support_hint": support_hint,
            "conn_bg": conn_bg,
            "hard_neg": hard_neg,
            "retained_response64": retained_response64,
            "contradict": contradict,

            # unified semantic interface
            "support": support_hint,
            "clutter": clutter,
            "rejection": rejection,
            "uncertainty": unc,
            "line": line,
            "coh": coh,
        }
        return z, pack


class EncoderStage3(nn.Module):
    def __init__(self, c: int, fs_channels: int = 5, fb_channels: int = 5,
                 d_state: int = 16, expand: int = 2):
        super().__init__()
        self.fs_field_proj = nn.Conv2d(fs_channels, c, 1, bias=False)
        self.fb_field_proj = nn.Conv2d(fb_channels, c, 1, bias=False)
        self.prior_proj = nn.Conv2d(3, c, 1, bias=False)
        self.mp_mamba = MPMambaBlock(c, d_state=d_state, expand=expand)
        self.background_token_reader = SceneTokenReader(c)
        self.structure_token_reader = SceneTokenReader(c)
        self.source_token_reader = SceneTokenReader(c)
        self.fuse = nn.Sequential(ConvGNAct(c * 6, c, 3, 1), ResidualBlock(c))
        self.pos_res = nn.Conv2d(c, 1, 1)
        self.neg_res = nn.Conv2d(c, 1, 1)
        self.relation_logits = nn.Conv2d(c, 1, 1)
        self.bnd_logits = nn.Conv2d(c, 1, 1)
        self.det_head = nn.Conv2d(c, 1, 1)
        self.bnd_gate_head = nn.Conv2d(c, 1, 1)
        self.reset_head = nn.Conv2d(c, 1, 1)
        self.rel_head = nn.Conv2d(c, 1, 1)
        self.unc_head = nn.Conv2d(c, 1, 1)
        self.progress = 0.0

    def set_progress(self, epoch: int, max_epoch: int):
        self.progress = min(1.0, float(epoch) / float(max(1, max_epoch)))

    def forward(
        self,
        x: torch.Tensor,
        fs_field: torch.Tensor,
        fb_field: torch.Tensor,
        background_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        e_pos_prior: torch.Tensor,
        e_neg_prior: torch.Tensor,
        u_prior: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ramp = 0.15 + 0.85 * self.progress
        ps = self.fs_field_proj(fs_field)
        pc = self.fb_field_proj(fb_field)
        prior = self.prior_proj(torch.cat([e_pos_prior, e_neg_prior, u_prior], dim=1))
        x0 = x + (0.10 + 0.15 * ramp) * ps - (0.05 + 0.10 * ramp) * pc + prior
        memory = torch.cat([background_tokens, structure_tokens, source_tokens], dim=1)
        y, memory = self.mp_mamba(x0, memory)
        n_background, n_structure, n_source = background_tokens.shape[1], structure_tokens.shape[1], source_tokens.shape[1]
        background_tokens = memory[:, :n_background]
        structure_tokens = memory[:, n_background : n_background + n_structure]
        source_tokens = memory[:, n_background + n_structure : n_background + n_structure + n_source]
        background_read = self.background_token_reader(x + y, background_tokens)
        structure_read = self.structure_token_reader(x + y, structure_tokens)
        source_read = self.source_token_reader(x + y, source_tokens)
        pos_residual = (x + y + source_read - background_read).abs()
        neg_residual = (x + y + structure_read - source_read).abs()
        z = self.fuse(torch.cat([x + y, background_read, structure_read, source_read, pos_residual, neg_residual], dim=1))
        rel_local = torch.sigmoid(self.rel_head(z))
        reset_gate = torch.sigmoid(self.reset_head(z + neg_residual - pos_residual))
        pos_local = torch.sigmoid(self.pos_res(z + pos_residual))
        neg_local = torch.sigmoid(self.neg_res(z + neg_residual))
        reliability = (
            (1.0 - (0.20 + 0.25 * ramp) * reset_gate)
            * (1.0 - 0.35 * u_prior)
            * rel_local
        )
        e_pos = reliability * e_pos_prior + (1.0 - reliability) * pos_local
        e_neg = (1.0 - 0.35 * reliability) * e_neg_prior + reliability * neg_local
        relation_logits = self.relation_logits(z + pos_residual - neg_residual + (e_pos - e_neg))
        keep_map = torch.sigmoid(relation_logits)
        bnd_logits = self.bnd_logits(z + pos_residual)
        bnd = torch.sigmoid(bnd_logits)
        neg_suppress = 0.25 + 0.50 * ramp
        reset_suppress = 0.08 + 0.27 * ramp
        g_det = (
            torch.sigmoid(self.det_head(z))
            * (1.0 - neg_suppress * e_neg)
            * (0.78 + 0.22 * e_pos)
            * (1.0 - reset_suppress * reset_gate)
        ).clamp(0.0, 1.0)
        g_bnd = (
            torch.sigmoid(self.bnd_gate_head(z))
            * (0.20 + 0.80 * keep_map)
            * (1.0 - 0.35 * e_neg)
        ).clamp(0.0, 1.0)
        unc = torch.sigmoid(self.unc_head((pos_residual - neg_residual).abs()))
        support_refine = (
            keep_map * (1.0 - e_neg) * (0.55 + 0.45 * e_pos)
        ).clamp(0.0, 1.0)
        risk_refine = (
            e_neg * (0.60 + 0.40 * reset_gate) * (0.70 + 0.30 * unc)
        ).clamp(0.0, 1.0)
        coh = fb_field[:, 2:3].clamp(0.0, 1.0)
        line = fb_field[:, 3:4].clamp(0.0, 1.0)
        clutter = torch.clamp(0.55 * e_neg + 0.25 * reset_gate + 0.20 * unc, 0.0, 1.0)
        rejection = torch.clamp(0.55 * risk_refine + 0.25 * reset_gate + 0.20 * bnd * (1.0 - support_refine), 0.0, 1.0)
        pack = {
            'relation_logits': relation_logits,
            'keep_map': keep_map,
            'bnd_logits': bnd_logits,
            'bnd': bnd,
            'g_det': g_det,
            'g_bnd': g_bnd,
            'reset_gate': reset_gate,
            'reliability': rel_local,
            'contrast_pos': pos_local,
            'contrast_neg': neg_local,
            'contrast_residual': (pos_local - neg_local).clamp(-1.0, 1.0),
            'e_pos': e_pos,
            'e_neg': e_neg,
            'u': unc,
            'support_refine': support_refine,
            'risk_refine': risk_refine,
            'scene_tokens': memory,
            'background_tokens': background_tokens,
            'structure_tokens': structure_tokens,
            'source_tokens': source_tokens,
            'background_read': background_read,
            'structure_read': structure_read,
            'source_read': source_read,
            'g': g_det,

            # unified semantic interface
            'support': support_refine,
            'clutter': clutter,
            'rejection': rejection,
            'uncertainty': unc,
            'line': line,
            'coh': coh,
        }
        return z, pack


class EncoderStage4(nn.Module):
    def __init__(self, c: int, fb_channels: int = 5, ff_channels: int = 5, d_state: int = 16, expand: int = 2):
        super().__init__()
        self.fb_field_proj = nn.Conv2d(fb_channels, c, 1, bias=False)
        self.ff_field_proj = nn.Conv2d(ff_channels, c, 1, bias=False)
        self.prior_proj = nn.Conv2d(3, c, 1, bias=False)

        self.mp_mamba = MPMambaBlock(
            c,
            d_state=d_state,
            expand=expand,
            orders=('row', 'col', 'diag', 'anti', 'snake')
        )

        self.background_token_reader = SceneTokenReader(c)
        self.structure_token_reader = SceneTokenReader(c)
        self.source_token_reader = SceneTokenReader(c)

        self.fuse = nn.Sequential(
            ConvGNAct(c * 6, c, 3, 1),
            ResidualBlock(c)
        )

        self.scene_interference_head = nn.Conv2d(c, 1, 1)
        self.repeat_bg = nn.Conv2d(c, 1, 1)
        self.interference_head = nn.Conv2d(c, 1, 1)
        self.boundary_reject = nn.Conv2d(c, 1, 1)
        self.det_head = nn.Conv2d(c, 1, 1)

        # Sparse rejection learns hard structured-interference patterns rather than broad background regions.
        self.sparse_rejection = nn.Sequential(
            ConvGNAct(c + 5, c, 3, 1),
            nn.Conv2d(c, 1, 1)
        )

        self.progress = 0.0

    def set_progress(self, epoch: int, max_epoch: int):
        self.progress = min(1.0, float(epoch) / float(max(1, max_epoch)))

    def forward(
        self,
        x: torch.Tensor,
        fb_field: torch.Tensor,
        ff_field: torch.Tensor,
        background_tokens: torch.Tensor,
        structure_tokens: torch.Tensor,
        source_tokens: torch.Tensor,
        e_neg_prior: torch.Tensor,
        keep_prior: torch.Tensor,
        u_prior: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        ramp = 0.15 + 0.85 * self.progress

        pc = self.fb_field_proj(fb_field)
        pf = self.ff_field_proj(ff_field)
        prior = self.prior_proj(torch.cat([e_neg_prior, keep_prior, u_prior], dim=1))

        x0 = x + (0.15 + 0.20 * ramp) * pc + (0.10 + 0.15 * ramp) * pf + prior

        memory = torch.cat([background_tokens, structure_tokens, source_tokens], dim=1)
        y, memory = self.mp_mamba(x0, memory)

        n_background, n_structure, n_source = background_tokens.shape[1], structure_tokens.shape[1], source_tokens.shape[1]
        background_tokens = memory[:, :n_background]
        structure_tokens = memory[:, n_background:n_background + n_structure]
        source_tokens = memory[:, n_background + n_structure:n_background + n_structure + n_source]

        background_read = self.background_token_reader(x + y, background_tokens)
        structure_read = self.structure_token_reader(x + y, structure_tokens)
        source_read = self.source_token_reader(x + y, source_tokens)

        z = self.fuse(torch.cat([x + y, background_read, structure_read, source_read, pc, pf], dim=1))

        repeat_bg4 = torch.sigmoid(self.repeat_bg(z + background_read + 0.20 * pc))
        scene_interference4 = torch.sigmoid(self.scene_interference_head(z + structure_read + 0.25 * pc - 0.20 * source_read))
        interference4 = torch.sigmoid(self.interference_head(z + 0.30 * pf + 0.20 * scene_interference4 + 0.18 * structure_read - 0.12 * source_read))
        boundary_reject4 = torch.sigmoid(self.boundary_reject(z + 0.30 * interference4 + 0.25 * keep_prior + 0.18 * pc))

        # Expose line/coherence responses for diagnostics and supervision.
        coh = fb_field[:, 2:3].clamp(0.0, 1.0)
        line = fb_field[:, 3:4].clamp(0.0, 1.0)

        contradict4 = (keep_prior - (1.0 - e_neg_prior)).abs().clamp(0.0, 1.0)

        sparse_rejection4 = torch.sigmoid(
            self.sparse_rejection(torch.cat([z, interference4, scene_interference4, boundary_reject4, contradict4, line], dim=1))
        )
        sparse_rejection4 = (
            sparse_rejection4 *
            torch.sigmoid(
                6.0 * (
                    sparse_rejection4 -
                    F.avg_pool2d(sparse_rejection4, kernel_size=5, stride=1, padding=2)
                )
            )
        ).clamp(0.0, 1.0)

        g_det = (
            torch.sigmoid(self.det_head(z)) *
            (1.0 - 0.18 * scene_interference4) *
            (1.0 - 0.16 * interference4) *
            (1.0 - 0.18 * sparse_rejection4) *
            (0.75 + 0.25 * keep_prior)
        ).clamp(0.0, 1.0)

        rejection_response = (
            0.40 * sparse_rejection4 +
            0.25 * interference4 +
            0.20 * scene_interference4 +
            0.15 * boundary_reject4
        ).clamp(0.0, 1.0)

        support = (
            keep_prior *
            (1.0 - 0.20 * sparse_rejection4) *
            (1.0 - 0.20 * interference4) *
            (1.0 - 0.18 * scene_interference4)
        ).clamp(0.0, 1.0)

        clutter = (
            0.35 * repeat_bg4 +
            0.20 * scene_interference4 +
            0.15 * interference4 +
            0.30 * sparse_rejection4
        ).clamp(0.0, 1.0)

        uncertainty = (
            0.35 * scene_interference4 +
            0.30 * (1.0 - keep_prior) +
            0.35 * contradict4
        ).clamp(0.0, 1.0)

        pack = {
            'scene_interference': scene_interference4,
            'repeat_bg4': repeat_bg4,
            'interference4': interference4,
            'boundary_reject4': boundary_reject4,
            'sparse_rejection4': sparse_rejection4,
            'contradict4': contradict4,
            'rejection_response': rejection_response,
            'g_det': g_det,
            'g': g_det,
            'scene_tokens': memory,
            'background_tokens': background_tokens,
            'structure_tokens': structure_tokens,
            'source_tokens': source_tokens,
            'background_read': background_read,
            'structure_read': structure_read,
            'source_read': source_read,

            # Keep field names used by the current decoder and analysis code.
            'coarse_clutter': repeat_bg4,
            'interference_basin': interference4,
            'reject_score': boundary_reject4,

            # unified semantic interface
            'support': support,
            'clutter': clutter,
            'rejection': rejection_response,
            'uncertainty': uncertainty,
            'line': line,
            'coh': coh,
        }
        return z, pack

class ResponseConditionedDecoder(nn.Module):
    def __init__(self, dims: Tuple[int, int, int, int], detail_dims: Tuple[int, int] = (32, 24)):
        super().__init__()
        c1, c2, c3, c4 = dims
        d_half, d_full = detail_dims

        self.up_fuse3 = UpFuseBlock(c4, c3, c3, field_c=5)
        self.up_fuse2 = UpFuseBlock(c3, c2, c2, field_c=5)
        self.up_fuse1 = UpFuseBlock(c2, c1, c1, field_c=5)

        self.to256 = nn.Sequential(
            ConvGNAct(c1 + d_half + 6, c1, 3, 1),
            ResidualBlock(c1)
        )
        self.to512 = nn.Sequential(
            ConvGNAct(c1 + d_full + 7, c1 // 2, 3, 1),
            ResidualBlock(c1 // 2)
        )

        self.src256 = nn.Conv2d(c1, 1, 1)
        self.edge256 = nn.Conv2d(c1, 1, 1)
        self.sdf256 = nn.Conv2d(c1, 1, 1)
        self.conf256 = nn.Conv2d(c1, 1, 1)
        self.support256 = nn.Conv2d(c1, 1, 1)
        self.center256 = nn.Conv2d(c1, 1, 1)
        self.interference256_head = nn.Conv2d(c1, 1, 1)

        self.src512 = nn.Conv2d(c1 // 2, 1, 1)
        self.edge512 = nn.Conv2d(c1 // 2, 1, 1)
        self.sdf512 = nn.Conv2d(c1 // 2, 1, 1)
        self.conf512 = nn.Conv2d(c1 // 2, 1, 1)
        self.support512 = nn.Conv2d(c1 // 2, 1, 1)
        self.center512 = nn.Conv2d(c1 // 2, 1, 1)
        self.interference512_head = nn.Conv2d(c1 // 2, 1, 1)
        self.boundary_rejection512_head = nn.Conv2d(c1 // 2, 1, 1)

        self.retained_candidate512_head = nn.Conv2d(c1 // 2, 1, 1)

        self.risk_refine512 = nn.Sequential(
            ConvGNAct(c1 // 2 + 5, c1 // 2, 3, 1),
            ResidualBlock(c1 // 2)
        )
        self.rejection_refine512 = nn.Sequential(
            ConvGNAct(c1 // 2 + 9, c1 // 2, 3, 1),
            ResidualBlock(c1 // 2)
        )
        self.risk512_head = nn.Conv2d(c1 // 2, 1, 1)
        self.rejection512_head = nn.Conv2d(c1 // 2, 1, 1)

        self.gamma_sup256 = nn.Parameter(torch.tensor(0.45))
        self.beta_bnd256 = nn.Parameter(torch.tensor(0.10))

        self.gamma_sup512 = nn.Parameter(torch.tensor(0.58))
        self.gamma_retention512 = nn.Parameter(torch.tensor(0.28))
        self.alpha_rejection512 = nn.Parameter(torch.tensor(0.52))
        self.alpha_risk512 = nn.Parameter(torch.tensor(0.26))
        self.alpha_ring512 = nn.Parameter(torch.tensor(0.18))
        self.beta_bnd512 = nn.Parameter(torch.tensor(0.12))
        self.progress = 0.0

    def set_progress(self, epoch: int, max_epoch: int):
        self.progress = min(1.0, float(epoch) / float(max(1, max_epoch)))

    @staticmethod
    def _resize(x: torch.Tensor, size_hw: Tuple[int, int]) -> torch.Tensor:
        return F.interpolate(x, size=size_hw, mode="bilinear", align_corners=False)

    @staticmethod
    def _field_stack(
        a: torch.Tensor, b: torch.Tensor, c: torch.Tensor,
        d: torch.Tensor, e: torch.Tensor,
        size_hw: Tuple[int, int]
    ) -> torch.Tensor:
        maps = [a, b, c, d, e]
        maps = [F.interpolate(m, size=size_hw, mode="bilinear", align_corners=False) for m in maps]
        return torch.cat(maps, dim=1)

    def forward(
        self,
        s1: torch.Tensor, s2: torch.Tensor, s3: torch.Tensor, s4: torch.Tensor,
        detail_half: torch.Tensor, detail_full: torch.Tensor,
        p1: Dict[str, torch.Tensor], p2: Dict[str, torch.Tensor],
        p3: Dict[str, torch.Tensor], p4: Dict[str, torch.Tensor],
        recon_err_full: torch.Tensor,
        point_prior_full: torch.Tensor,
        tiny_prior_full: torch.Tensor,
        sparse_full: torch.Tensor,
        veil_full: torch.Tensor,
        retention_prior_full: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:

        # -------- top-down --------
        pos128 = (torch.sigmoid(p1["source_logit"]) * p1["g_det"]).clamp(0.0, 1.0)
        neg128 = (
            0.40 * torch.sigmoid(p1["nuis_logit"]) +
            0.30 * p1["clutter_seed"] +
            0.15 * p1["line_reject"] +
            0.15 * p1["interference_seed"]
        ).clamp(0.0, 1.0)

        structure64 = p2["clutter_prob"]
        conn64 = p2.get("conn_bg", torch.zeros_like(structure64))

        pos32 = p3["e_pos"] * p3["g_det"]
        neg32 = p3["e_neg"] * (0.60 + 0.40 * p3["reset_gate"])
        rejection32 = p3["rejection"]
        support32 = p3.get("support_refine", p3["keep_map"])

        field3 = self._field_stack(
            self._resize(pos128, s3.shape[-2:]),
            self._resize(neg128, s3.shape[-2:]),
            support32,
            rejection32,
            self._resize(p4["scene_interference"], s3.shape[-2:]),
            s3.shape[-2:]
        )
        x = self.up_fuse3(s4, s3, field3)

        field2 = self._field_stack(
            self._resize(pos128, s2.shape[-2:]),
            structure64,
            self._resize(support32, s2.shape[-2:]),
            self._resize(rejection32, s2.shape[-2:]),
            self._resize(conn64, s2.shape[-2:]),
            s2.shape[-2:]
        )
        x = self.up_fuse2(x, s2, field2)

        field1 = self._field_stack(
            pos128,
            self._resize(structure64, s1.shape[-2:]),
            self._resize(support32, s1.shape[-2:]),
            self._resize(rejection32, s1.shape[-2:]),
            self._resize(conn64, s1.shape[-2:]),
            s1.shape[-2:]
        )
        x = self.up_fuse1(x, s1, field1)

        # -------- 256 --------
        x256 = F.interpolate(x, scale_factor=2, mode="bilinear", align_corners=False)

        pos256 = (
            0.34 * self._resize(pos128, x256.shape[-2:]) +
            0.22 * self._resize(pos32, x256.shape[-2:]) +
            0.18 * self._resize(support32, x256.shape[-2:]) +
            0.14 * self._resize(p1["center_seed"], x256.shape[-2:]) +
            0.12 * self._resize(p1["tiny_target_prior"], x256.shape[-2:])
        ).clamp(0.0, 1.0)

        neg256 = (
            0.34 * self._resize(neg128, x256.shape[-2:]) +
            0.22 * self._resize(structure64, x256.shape[-2:]) +
            0.18 * self._resize(neg32, x256.shape[-2:]) +
            0.12 * self._resize(conn64, x256.shape[-2:]) +
            0.14 * self._resize(p4["scene_interference"], x256.shape[-2:])
        ).clamp(0.0, 1.0)

        boundary256 = (
            0.40 * self._resize(torch.sigmoid(p1["bnd_logits"]), x256.shape[-2:]) +
            0.30 * self._resize(p3["bnd"], x256.shape[-2:]) +
            0.15 * self._resize(p1["support_seed"], x256.shape[-2:]) +
            0.15 * self._resize(p1["center_seed"], x256.shape[-2:])
        ).clamp(0.0, 1.0)

        point256 = self._resize(point_prior_full, x256.shape[-2:])
        tiny256 = self._resize(tiny_prior_full, x256.shape[-2:])

        support_seed256 = (
            0.22 * self._resize(p1["cand_map"], x256.shape[-2:]) +
            0.20 * self._resize(p3["keep_map"], x256.shape[-2:]) +
            0.16 * boundary256 +
            0.14 * point256 +
            0.14 * self._resize(p1["center_seed"], x256.shape[-2:]) +
            0.14 * tiny256
        ).clamp(0.0, 1.0)

        interference_prior256 = (
            0.45 * self._resize(p3["rejection"], x256.shape[-2:]) +
            0.25 * self._resize(p4["interference4"], x256.shape[-2:]) +
            0.15 * self._resize(conn64, x256.shape[-2:]) +
            0.15 * self._resize(p4["scene_interference"], x256.shape[-2:])
        ).clamp(0.0, 1.0)

        field256 = torch.cat([
            pos256,
            neg256,
            boundary256,
            support_seed256,
            interference_prior256,
            0.55 * point256 + 0.45 * tiny256
        ], dim=1)

        x256 = self.to256(torch.cat([x256, detail_half, field256], dim=1))

        src256 = self.src256(x256)
        edge256 = self.edge256(x256 + boundary256)
        sdf256 = self.sdf256(x256)
        conf256 = torch.sigmoid(self.conf256(x256))
        support256 = torch.sigmoid(
            self.support256(x256 + 0.35 * support_seed256 + 0.10 * point256 + 0.20 * tiny256)
        )
        center256 = torch.sigmoid(
            self.center256(x256 + 0.25 * point256 + 0.25 * tiny256)
        )
        interference256 = torch.sigmoid(
            self.interference256_head(x256 + 0.30 * interference_prior256)
        )

        band256 = (
            torch.sigmoid(edge256) *
            torch.sigmoid(4.0 * boundary256) *
            (0.45 * support256 + 0.30 * center256 + 0.25 * conf256)
        ).clamp(0.0, 1.0)

        logit256 = (
            src256 +
            self.gamma_sup256.clamp(0.0, 1.5) *
            (0.54 * support256 + 0.24 * center256 + 0.12 * conf256 + 0.10 * tiny256)
            - 0.22 * neg256
            - 0.16 * interference256
            + self.beta_bnd256.clamp(0.0, 0.5) * band256
        )

        # -------- 512 --------
        x512 = F.interpolate(x256, scale_factor=2, mode="bilinear", align_corners=False)

        recon512 = self._resize(recon_err_full, x512.shape[-2:])
        sparse512 = self._resize(sparse_full, x512.shape[-2:])
        veil512 = self._resize(veil_full, x512.shape[-2:])
        point512 = self._resize(point_prior_full, x512.shape[-2:])
        tiny512 = self._resize(tiny_prior_full, x512.shape[-2:])

        if retention_prior_full is None:
            retention_prior512 = tiny512
        else:
            retention_prior512 = self._resize(retention_prior_full, x512.shape[-2:])

        conf256_up = self._resize(conf256, x512.shape[-2:])
        support256_up = self._resize(support256, x512.shape[-2:])
        center256_up = self._resize(center256, x512.shape[-2:])
        logit256_prob = self._resize(torch.sigmoid(logit256).detach(), x512.shape[-2:])

        pos512 = (
            0.24 * support256_up +
            0.18 * center256_up +
            0.14 * conf256_up * logit256_prob +
            0.12 * self._resize(pos256, x512.shape[-2:]) +
            0.10 * self._resize(support32, x512.shape[-2:]) +
            0.10 * tiny512 +
            0.12 * retention_prior512
        ).clamp(0.0, 1.0)

        interference4_full = self._resize(p4["interference4"], x512.shape[-2:])
        scene4 = self._resize(p4["scene_interference"], x512.shape[-2:])
        reject4 = self._resize(p4["boundary_reject4"], x512.shape[-2:])
        conn512 = self._resize(conn64, x512.shape[-2:])
        sparse_rejection4 = self._resize(p4.get("sparse_rejection4", p4["rejection"]), x512.shape[-2:])

        boundary512 = (
            0.26 * self._resize(torch.sigmoid(edge256), x512.shape[-2:]) +
            0.18 * self._resize(boundary256, x512.shape[-2:]) +
            0.16 * self._resize(p3["bnd"], x512.shape[-2:]) +
            0.14 * reject4 +
            0.10 * pos512 +
            0.08 * conn512 +
            0.08 * sparse_rejection4
        ).clamp(0.0, 1.0)

        support_seed512 = (
            0.20 * support256_up +
            0.12 * center256_up +
            0.12 * logit256_prob +
            0.10 * point512 +
            0.14 * tiny512 +
            0.14 * retention_prior512 +
            0.10 * pos512 +
            0.08 * boundary512
        ).clamp(0.0, 1.0)

        field512 = torch.cat([
            pos512,
            boundary512,
            support_seed512,
            interference4_full,
            scene4,
            recon512,
            0.35 * point512 + 0.35 * tiny512 + 0.30 * retention_prior512
        ], dim=1)

        x512 = self.to512(torch.cat([x512, detail_full, field512], dim=1))

        src512 = self.src512(x512)
        edge512 = self.edge512(x512 + boundary512)
        sdf512 = self.sdf512(x512)
        conf512 = torch.sigmoid(self.conf512(x512))

        support512 = torch.sigmoid(
            self.support512(
                x512
                + 0.28 * support_seed512
                + 0.16 * sparse512
                + 0.10 * point512
                + 0.16 * tiny512
                + 0.12 * retention_prior512
            )
        )
        center512 = torch.sigmoid(
            self.center512(
                x512
                + 0.20 * point512
                + 0.20 * tiny512
                + 0.16 * support_seed512
                + 0.12 * retention_prior512
            )
        )
        interference512 = torch.sigmoid(
            self.interference512_head(
                x512 + 0.26 * interference4_full + 0.18 * recon512 + 0.08 * veil512 + 0.08 * scene4 + 0.10 * conn512
            )
        )
        boundary_rejection512 = torch.sigmoid(
            self.boundary_rejection512_head(
                x512 + 0.28 * boundary512 + 0.24 * reject4 + 0.15 * interference512 + 0.10 * conn512
            )
        )

        retained_candidate512 = torch.sigmoid(
            self.retained_candidate512_head(
                x512 + 0.30 * support_seed512 + 0.20 * point512 + 0.22 * tiny512 + 0.18 * retention_prior512 - 0.08 * interference4_full
            )
        )
        preservation_floor512 = (
            0.40 * support_seed512 +
            0.22 * point512 +
            0.22 * tiny512 +
            0.16 * retention_prior512
        ).clamp(0.0, 1.0)
        retained_candidate512 = torch.maximum(
            retained_candidate512,
            0.65 * preservation_floor512 *
            torch.sigmoid(4.0 * (0.60 - interference512)) *
            torch.sigmoid(4.0 * (0.58 - conn512))
        )

        contradict512 = (
            (support512 - center512).abs() +
            0.5 * (support512 - conf512).abs()
        ).clamp(0.0, 1.0)

        risk_feat512 = self.risk_refine512(torch.cat([
            x512,
            interference512,
            scene4,
            conn512,
            contradict512,
            1.0 - support512
        ], dim=1))
        risk512 = torch.sigmoid(
            self.risk512_head(
                risk_feat512 + 0.18 * interference512 + 0.14 * conn512 + 0.14 * contradict512
            )
        )

        rejection_feat512 = self.rejection_refine512(torch.cat([
            x512,
            support512,
            center512,
            conf512,
            interference512,
            boundary_rejection512,
            risk512,
            conn512,
            sparse_rejection4,
            contradict512
        ], dim=1))
        rejection512 = torch.sigmoid(
            self.rejection512_head(
                rejection_feat512
                + 0.16 * contradict512
                + 0.16 * boundary_rejection512
                + 0.14 * interference512
                + 0.12 * sparse_rejection4
                - 0.10 * center512
            )
        )

        band512 = (
            torch.sigmoid(edge512) *
            torch.sigmoid(4.0 * boundary512) *
            (
                0.36 * support512 +
                0.24 * center512 +
                0.15 * conf512 +
                0.15 * retained_candidate512 +
                0.10 * (1.0 - rejection512)
            )
        ).clamp(0.0, 1.0)

        outer_ring512 = (
            torch.sigmoid(edge512) *
            torch.sigmoid(4.0 * boundary512) *
            (1.0 - 0.55 * support512)
        ).clamp(0.0, 1.0)

        ring_response512 = (
            outer_ring512 *
            (0.55 * rejection512 + 0.45 * risk512) *
            (0.70 + 0.30 * conn512)
        ).clamp(0.0, 1.0)

        positive512 = (
            self.gamma_sup512.clamp(0.0, 1.5) *
            (0.36 * support512 + 0.22 * center512 + 0.12 * conf512 + 0.16 * tiny512 + 0.14 * point512)
            + self.gamma_retention512.clamp(0.0, 1.0) * retained_candidate512
        )

        protection_response512 = (
            0.42 * support512 +
            0.22 * center512 +
            0.16 * conf512 +
            0.20 * retained_candidate512
        ).clamp(0.0, 1.0)

        rejection_penalty = self.alpha_rejection512.clamp(0.0, 1.2) * rejection512 * (1.0 - 0.60 * protection_response512)
        risk_penalty = self.alpha_risk512.clamp(0.0, 1.0) * risk512 * (1.0 - 0.45 * protection_response512)
        ring_penalty = self.alpha_ring512.clamp(0.0, 0.6) * ring_response512

        logit512 = (
            src512 +
            positive512 -
            rejection_penalty -
            risk_penalty -
            ring_penalty +
            self.beta_bnd512.clamp(0.0, 0.5) * band512
        )

        return {
            "logits256": logit256,
            "edge256": edge256,
            "sdf256": sdf256,
            "conf256": conf256,
            "support256": support256,
            "center256": center256,
            "interference256": interference256,

            "logits512": logit512,
            "edge512": edge512,
            "sdf512": sdf512,
            "conf512": conf512,
            "support512": support512,
            "center512": center512,
            "boundary_rejection512": boundary_rejection512,
            "interference512": interference512,
            "risk512": risk512,
            "rejection512": rejection512,
            "retained_candidate512": retained_candidate512,
            "ring_response512": ring_response512,
        }


class DRCMamba(nn.Module):
    def __init__(
        self,
        in_ch: int = 1,
        dims: Tuple[int, int, int, int] = (64, 128, 256, 256),
        d_state: int = 16,
        expand: int = 2
    ):
        super().__init__()
        c1, c2, c3, c4 = dims
        fs_channels, fb_channels, ff_channels = 5, 5, 5

        self.response_decomposition = ResponseDecompositionModule()
        self.cp_branch = CandidatePreservationBranch(
            fs_channels=fs_channels, fb_channels=fb_channels, ff_channels=ff_channels,
            detail_dims=(24, 32, 48), retention_hidden=24
        )

        self.stem_half = nn.Sequential(
            ConvGNAct(in_ch + 10, 32, 5, 2),
            ResidualBlock(32)
        )
        self.stem_quarter = nn.Sequential(
            Downsample(32, c1),
            ResidualBlock(c1)
        )
        self.detail_quarter_proj = nn.Conv2d(48, c1, 1, bias=False)

        self.stage1 = EncoderStage1(
            c1,
            detail_c=48,
            fs_channels=fs_channels,
            fb_channels=fb_channels,
            ff_channels=ff_channels
        )
        self.down1 = Downsample(c1, c2)

        # Context stages.
        self.stage2 = EncoderStage2(
            c2,
            fs_channels=fs_channels,
            fb_channels=fb_channels,
            background_tokens=6,
            structure_tokens=6,
            source_tokens=4,
            d_state=d_state,
            expand=expand
        )

        self.down2 = Downsample(c2, c3)
        self.background32_proj = nn.Linear(c2, c3)
        self.structure32_proj = nn.Linear(c2, c3)
        self.source32_proj = nn.Linear(c2, c3)

        self.stage3 = EncoderStage3(
            c3,
            fs_channels=fs_channels,
            fb_channels=fb_channels,
            d_state=d_state,
            expand=expand
        )

        self.down3 = Downsample(c3, c4)
        self.background16_proj = nn.Linear(c3, c4)
        self.structure16_proj = nn.Linear(c3, c4)
        self.source16_proj = nn.Linear(c3, c4)

        self.stage4 = EncoderStage4(
            c4,
            fb_channels=fb_channels,
            ff_channels=ff_channels,
            d_state=d_state,
            expand=expand
        )

        self.rcd = ResponseConditionedDecoder(dims=dims, detail_dims=(32, 24))

    def forward(self, x: torch.Tensor) -> Dict[str, Any]:
        if x.shape[1] != 1:
            x = x.mean(dim=1, keepdim=True)

        response_fields = self.response_decomposition(x)
        cp_detail, cp_retention = self.cp_branch(x, response_fields)

        x_half = self.stem_half(torch.cat([x, response_fields["full"]["base"]], dim=1))
        x_quarter = self.stem_quarter(x_half)
        x_quarter = x_quarter + 0.05 * self.detail_quarter_proj(cp_detail["quarter"])

        point_q = F.interpolate(cp_retention["point_prior"], size=x_quarter.shape[-2:], mode="bilinear", align_corners=False)
        center_q = F.interpolate(cp_retention["center_prior"], size=x_quarter.shape[-2:], mode="bilinear",
                                 align_corners=False)
        tiny_q = F.interpolate(cp_retention["tiny_target_map"], size=x_quarter.shape[-2:], mode="bilinear",
                               align_corners=False)
        retention_q = F.interpolate(cp_retention["retention_gate"], size=x_quarter.shape[-2:], mode="bilinear", align_corners=False)

        # stage1
        s1, p1 = self.stage1(
            x_quarter,
            cp_detail["fs_quarter"],
            cp_detail["fb_quarter"],
            response_fields["quarter"]["fs"],
            response_fields["quarter"]["fb"],
            response_fields["quarter"]["ff"],
            point_q,
            center_q,
            tiny_q,
            retention_q
        )

        # stage2
        s2_in = self.down1(s1)

        # Fuse the retained-candidate response at Stage 1 before downsampling to Stage 2.
        retention_p1 = F.interpolate(
            cp_retention["retention_response"],
            size=p1["g_det"].shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        source_prior64 = F.interpolate(
            (
                    0.26 * p1["g_det"] +
                    0.14 * p1["cand_map"] +
                    0.14 * p1["support"] +
                    0.10 * p1["point_prior"] +
                    0.10 * p1["center_seed"] +
                    0.10 * p1["tiny_target_prior"] +
                    0.16 * retention_p1
            ).clamp(0.0, 1.0),
            size=s2_in.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        structure_hint64 = F.interpolate(
            (
                    0.34 * p1["clutter_seed"] +
                    0.20 * p1["line_reject"] +
                    0.20 * p1["risk_seed"] +
                    0.16 * p1["interference_seed"] +
                    0.10 * p1["contradict"]
            ).clamp(0.0, 1.0),
            size=s2_in.shape[-2:],
            mode="bilinear",
            align_corners=False
        )

        s2, p2 = self.stage2(
            s2_in,
            response_fields["eighth"]["fs"],
            response_fields["eighth"]["fb"],
            source_prior64,
            structure_hint64
        )

        # stage3
        s3_in = self.down2(s2)
        epos32 = F.interpolate(p2["e_pos"], size=s3_in.shape[-2:], mode="bilinear", align_corners=False)
        eneg32 = F.interpolate(p2["e_neg"], size=s3_in.shape[-2:], mode="bilinear", align_corners=False)
        u32 = F.interpolate(p2["u"], size=s3_in.shape[-2:], mode="bilinear", align_corners=False)

        background32 = self.background32_proj(p2["background_tokens"])
        structure32 = self.structure32_proj(p2["structure_tokens"])
        source32 = self.source32_proj(p2["source_tokens"])

        s3, p3 = self.stage3(
            s3_in,
            response_fields["sixteenth"]["fs"],
            response_fields["sixteenth"]["fb"],
            background32, structure32, source32,
            epos32, eneg32, u32
        )

        # stage4
        s4_in = self.down3(s3)
        eneg16 = F.interpolate(p3["e_neg"], size=s4_in.shape[-2:], mode="bilinear", align_corners=False)
        keep16 = F.interpolate(p3["keep_map"], size=s4_in.shape[-2:], mode="bilinear", align_corners=False)
        u16 = F.interpolate(p3["u"], size=s4_in.shape[-2:], mode="bilinear", align_corners=False)

        background16 = self.background16_proj(p3["background_tokens"])
        structure16 = self.structure16_proj(p3["structure_tokens"])
        source16 = self.source16_proj(p3["source_tokens"])

        s4, p4 = self.stage4(
            s4_in,
            response_fields["thirty_second"]["fb"],
            response_fields["thirty_second"]["ff"],
            background16, structure16, source16,
            eneg16, keep16, u16
        )

        retention_full = torch.maximum(
            cp_retention["retention_response"],
            F.interpolate(
                p2["retained_response64"],
                size=response_fields["full"]["recon_err"].shape[-2:],
                mode="bilinear",
                align_corners=False
            )
        )

        dec = self.rcd(
            s1, s2, s3, s4,
            cp_detail["half"], cp_detail["full"],
            p1, p2, p3, p4,
            recon_err_full=response_fields["full"]["recon_err"],
            point_prior_full=cp_retention["point_prior"],
            tiny_prior_full=cp_retention["tiny_target_map"],
            sparse_full=response_fields["full"]["sparse"],
            veil_full=response_fields["full"]["veil"],
            retention_prior_full=retention_full,
        )

        return {
            "mask": dec["logits512"],
            "edge": dec["edge512"],
            "aux1": dec["logits256"],
            "interference256": dec["interference256"],
            "sdf512": dec["sdf512"],
            "sdf256": dec["sdf256"],
            "conf512": dec["conf512"],
            "conf256": dec["conf256"],
            "support512": dec["support512"],
            "support256": dec["support256"],
            "center512": dec["center512"],
            "interference512": dec["interference512"],
            "boundary_rejection512": dec["boundary_rejection512"],
            "risk512": dec["risk512"],
            "rejection512": dec["rejection512"],
            "retained_candidate512": dec["retained_candidate512"],
            "ring_response512": dec["ring_response512"],
            "packs": {"p1": p1, "p2": p2, "p3": p3, "p4": p4},

            "bg_recon": response_fields["full"]["bg"],
            "bg_veil": response_fields["full"]["veil"],
            "bg_sparse": response_fields["full"]["sparse"],
            "bg_recon_err": response_fields["full"]["recon_err"],
            "bg_struct": response_fields["full"]["bg_struct"],
            "point_prior": cp_retention["point_prior"],
            "center_prior": cp_retention["center_prior"],
            "tiny_target_map": cp_retention["tiny_target_map"],
            "retention_response": cp_retention["retention_response"],
        }



def build_drc_mamba() -> DRCMamba:
    """Build the DRC-Mamba configuration used in the paper experiments."""
    return DRCMamba(in_ch=1, dims=(64, 128, 256, 256), d_state=16, expand=2)
