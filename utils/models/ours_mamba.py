# Copyright (c) 2023, Tri Dao, Albert Gu.

import math
import numpy as np
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from einops import rearrange, repeat

from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, mamba_inner_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None

try:
    from mamba_ssm.ops.triton.selective_state_update import selective_state_update
except ImportError:
    selective_state_update = None

try:
    from mamba_ssm.ops.triton.layer_norm import RMSNorm, layer_norm_fn, rms_norm_fn
except ImportError:
    RMSNorm, layer_norm_fn, rms_norm_fn = None, None, None


class Dual_Mamba(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        d_conv=4,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        conv_bias=True,
        bias=False,
        use_fast_path=True,  # Fused kernel options
        layer_idx=None,
        device=None,
        dtype=None,
        A_log=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx

        # self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.register_buffer(
            'in_proj_weight',
            torch.empty(self.d_inner * 2, self.d_model)
        )
        self.use_in_bias = bias
        if self.use_in_bias:
            self.register_buffer(
                'in_proj_bias',
                torch.empty(self.d_inner * 2)
            )
        else:
            self.register_buffer('in_proj_bias', None)

        self.conv1d = nn.Conv1d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=self.d_inner,
            padding=d_conv - 1,
            **factory_kwargs,
        )

        self.activation = "silu"
        self.act = nn.SiLU()

        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = self.dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(self.d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        # ====================
        if A_log is not None:
            assert A_log.shape == (self.d_inner, self.d_state), \
                f"A_log shape mismatch: expected {(self.d_inner, self.d_state)}, got {A_log.shape}"


            self.register_buffer('A_log', A_log.to(device=device, dtype=torch.float32))
        else:
            A = repeat(
                torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
                "n -> d n",
                d=self.d_inner,
            ).contiguous()
            A_log_init = torch.log(A)


            self.register_buffer('A_log', A_log_init)

        self.A_log._no_weight_decay = True
        # ====================
        self.aligner = ChannelDistributionAlignmentLoss()

        # D "skip" parameter
        self.D = nn.Parameter(torch.ones(self.d_inner, device=device))  # Keep in fp32
        self.D._no_weight_decay = True

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)

    def _invert_A(self, features):
        A_max = self.A_log.max(dim=0, keepdim=True)[0] # (1, d_state)
        A_min = self.A_log.min(dim=0, keepdim=True)[0] # (1, d_state)
        A_flipped = A_max + A_min - self.A_log # (d_inner, d_state)

        A = torch.exp(self.A_log)
        attention = A.norm(dim=1, p=2)  # (d_inner,)
        attention = attention / (attention.max() + 1e-8)

        features = features.detach().mean(dim=0)  # (d_inner,)
        features = F.softmax(features, dim=0)

        alpha = features * (1 - attention)  # (d_inner,)
        alpha = torch.clamp(alpha, 0, 1)
        alpha = alpha.unsqueeze(1)  # (d_inner, 1)

        self.A_log = (1 - alpha) * self.A_log + alpha * A_flipped

    def forward(self, hidden_states, T, H, W, inference_params=None):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        batch, seqlen, dim = hidden_states.shape

        conv_state, ssm_state = None, None
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.seqlen_offset > 0:
                # The states are updated inplace
                out, _, _ = self.step(hidden_states, conv_state, ssm_state)
                return out

        # We do matmul and transpose BLH -> HBL at the same time
        xz = rearrange(
            self.in_proj_weight @ rearrange(hidden_states, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=seqlen,
        )
        if self.in_proj_bias is not None:
            xz = xz + rearrange(self.in_proj_bias.to(dtype=xz.dtype), "d -> d 1")

        x, z = xz.chunk(2, dim=1)

        align_loss, channel_features = self.aligner(x, T, H, W, return_features=True)
        self._invert_A(channel_features)

        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # In the backward pass we write dx and dz next to each other to avoid torch.cat
        if self.use_fast_path and causal_conv1d_fn is not None and inference_params is None:  # Doesn't support outputting the states
            out = mamba_inner_fn(
                xz,
                self.conv1d.weight,
                self.conv1d.bias,
                self.x_proj.weight,
                self.dt_proj.weight,
                self.out_proj.weight,
                self.out_proj.bias,
                A,
                None,  # input-dependent B
                None,  # input-dependent C
                self.D.float(),
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
            )
        else:
            #x, z = xz.chunk(2, dim=1)
            # Compute short convolution
            if conv_state is not None:
                # If we just take x[:, :, -self.d_conv :], it will error if seqlen < self.d_conv
                # Instead F.pad will pad with zeros if seqlen < self.d_conv, and truncate otherwise.
                conv_state.copy_(F.pad(x, (self.d_conv - x.shape[-1], 0)))  # Update state (B D W)
            if causal_conv1d_fn is None:
                x = self.act(self.conv1d(x)[..., :seqlen])
            else:
                assert self.activation in ["silu", "swish"]
                x = causal_conv1d_fn(
                    x=x,
                    weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                    bias=self.conv1d.bias,
                    activation=self.activation,
                )

            # We're careful here about the layout, to avoid extra transposes.
            # We want dt to have d as the slowest moving dimension
            # and L as the fastest moving dimension, since those are what the ssm_scan kernel expects.
            x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))  # (bl d)

            dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
            dt = self.dt_proj.weight @ dt.t()
            dt = rearrange(dt, "d (b l) -> b d l", l=seqlen)
            B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
            assert self.activation in ["silu", "swish"]
            y = selective_scan_fn(
                x,
                dt,
                A,
                B,
                C,
                self.D.float(),
                z=z,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=ssm_state is not None,
            )
            if ssm_state is not None:
                y, last_state = y
                ssm_state.copy_(last_state)
            y = rearrange(y, "b d l -> b l d")
            out = self.out_proj(y)
        return out, align_loss

    def step(self, hidden_states, conv_state, ssm_state):
        dtype = hidden_states.dtype
        assert hidden_states.shape[1] == 1, "Only support decoding with 1 token at a time for now"
        #xz = self.in_proj(hidden_states.squeeze(1))  # (B 2D)
        xz = F.linear(hidden_states, self.in_proj_weight, self.in_proj_bias)  # (B 2D)
        x, z = xz.chunk(2, dim=-1)  # (B D)

        # Conv step
        if causal_conv1d_update is None:
            conv_state.copy_(torch.roll(conv_state, shifts=-1, dims=-1))  # Update state (B D W)
            conv_state[:, :, -1] = x
            x = torch.sum(conv_state * rearrange(self.conv1d.weight, "d 1 w -> d w"), dim=-1)  # (B D)
            if self.conv1d.bias is not None:
                x = x + self.conv1d.bias
            x = self.act(x).to(dtype=dtype)
        else:
            x = causal_conv1d_update(
                x,
                conv_state,
                rearrange(self.conv1d.weight, "d 1 w -> d w"),
                self.conv1d.bias,
                self.activation,
            )

        x_db = self.x_proj(x)  # (B dt_rank+2*d_state)

        dt, B, C = torch.split(x_db, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        # Don't add dt_bias here
        dt = F.linear(dt, self.dt_proj.weight)  # (B d_inner)
        A = -torch.exp(self.A_log.float())  # (d_inner, d_state)

        # SSM step
        if selective_state_update is None:
            # Discretize A and B
            dt = F.softplus(dt + self.dt_proj.bias.to(dtype=dt.dtype))
            dA = torch.exp(torch.einsum("bd,dn->bdn", dt, A))
            dB = torch.einsum("bd,bn->bdn", dt, B)
            ssm_state.copy_(ssm_state * dA + rearrange(x, "b d -> b d 1") * dB)
            y = torch.einsum("bdn,bn->bd", ssm_state.to(dtype), C)
            y = y + self.D.to(dtype) * x
            y = y * self.act(z)  # (B D)
        else:
            y = selective_state_update(
                ssm_state, x, dt, A, B, C, self.D, z=z, dt_bias=self.dt_proj.bias, dt_softplus=True
            )

        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state

    def allocate_inference_cache(self, batch_size, max_seqlen, dtype=None, **kwargs):
        device = self.out_proj.weight.device
        conv_dtype = self.conv1d.weight.dtype if dtype is None else dtype
        conv_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_conv, device=device, dtype=conv_dtype
        )
        ssm_dtype = self.dt_proj.weight.dtype if dtype is None else dtype
        # ssm_dtype = torch.float32
        ssm_state = torch.zeros(
            batch_size, self.d_model * self.expand, self.d_state, device=device, dtype=ssm_dtype
        )
        return conv_state, ssm_state

    def _get_states_from_cache(self, inference_params, batch_size, initialize_states=False):
        assert self.layer_idx is not None
        if self.layer_idx not in inference_params.key_value_memory_dict:
            batch_shape = (batch_size,)
            conv_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_conv,
                device=self.conv1d.weight.device,
                dtype=self.conv1d.weight.dtype,
            )
            ssm_state = torch.zeros(
                batch_size,
                self.d_model * self.expand,
                self.d_state,
                device=self.dt_proj.weight.device,
                dtype=self.dt_proj.weight.dtype,
                # dtype=torch.float32,
            )
            inference_params.key_value_memory_dict[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params.key_value_memory_dict[self.layer_idx]
            # TODO: What if batch size changes between generation, and we reuse the same states?
            if initialize_states:
                conv_state.zero_()
                ssm_state.zero_()
        return conv_state, ssm_state

    def set_A_log(self, A_log):
        """
        Args:
            A_log: shape (d_inner, d_state)
        """
        assert A_log.shape == (self.d_inner, self.d_state), \
            f"A_log shape mismatch: expected {(self.d_inner, self.d_state)}, got {A_log.shape}"

        # A_log = self._invert_values(A_log)

        # 更新 buffer
        self.A_log.copy_(A_log.to(device=self.A_log.device, dtype=torch.float32))

    def set_in_proj_weight(self, weight: torch.Tensor, bias: Optional[torch.Tensor] = None):
        """
        Args:
            weight: shape (d_inner * 2, d_model)
            bias: shape (d_inner * 2,) or None
        """
        expected_shape = (self.d_inner * 2, self.d_model)
        assert weight.shape == expected_shape, \
            f"Weight shape mismatch: expected {expected_shape}, got {weight.shape}"

        self.in_proj_weight.copy_(
            weight.to(device=self.in_proj_weight.device, dtype=self.in_proj_weight.dtype)
        )

        if bias is not None:
            if self.in_proj_bias is None:
                raise ValueError("Model was initialized without bias, cannot set bias")

            expected_bias_shape = (self.d_inner * 2,)
            assert bias.shape == expected_bias_shape, \
                f"Bias shape mismatch: expected {expected_bias_shape}, got {bias.shape}"

            self.in_proj_bias.copy_(
                bias.to(device=self.in_proj_bias.device, dtype=self.in_proj_bias.dtype)
            )



class ChannelDistributionAlignmentLoss(nn.Module):
    def __init__(
        self,
        metric='frequency',
        distance='cosine',   # 'cosine', 'l2', 'kl'
        temperature=0.1,
        normalize=True
    ):
        super().__init__()
        self.metric = metric
        self.distance = distance
        self.temperature = temperature
        self.normalize = normalize

    def compute_channel_features(self, x):
        """
        Args:
            x: (B, C, H, W) or (B*T, C, H, W)

        Returns:
            features: (B*T, C)
        """

        if self.metric == 'frequency':
            return self._frequency_spectrum_feature(x)
        else:
            raise ValueError(f"Unknown metric: {self.metric}")

    def _frequency_spectrum_feature(self, x):
        B, C, H, W = x.shape

        # FFT
        x_fft = torch.fft.rfft2(x, dim=(-2, -1))  # (B, C, H, W//2+1)
        magnitude = torch.sqrt(
            x_fft.real**2 + x_fft.imag**2 + 1e-8
        )
        magnitude = torch.fft.fftshift(magnitude, dim=(-2, -1))
        h_center, w_center = H // 2, magnitude.shape[-1] // 2
        h_grid = torch.arange(H, device=x.device).view(-1, 1)
        w_grid = torch.arange(magnitude.shape[-1], device=x.device).view(1, -1)

        freq_radius = torch.sqrt(
            ((h_grid - h_center) / H) ** 2 + 
            ((w_grid - w_center) / magnitude.shape[-1]) ** 2
        )  # (H, W//2+1)

        n_bands = 8
        features = []

        for i in range(n_bands):
            low = i / n_bands
            high = (i + 1) / n_bands

            mask = (freq_radius >= low) & (freq_radius < high)

            band_energy = (magnitude * mask.unsqueeze(0).unsqueeze(0)).sum(dim=(-2, -1))
            features.append(band_energy)

        features = torch.stack(features, dim=-1)  # (B, C, n_bands)

        features = features / (features.sum(dim=-1, keepdim=True) + 1e-8)

        high_freq_ratio = features[..., n_bands//2:].sum(dim=-1)  # (B, C)

        return high_freq_ratio


    def compute_pairwise_similarity(self, features):
        """
        Args:
            features: (B, C)
        
        Returns:
            similarity: (B, B)
        """
        B, _ = features.shape

        if self.normalize:
            features = F.normalize(features, p=2, dim=1)

        if self.distance == 'cosine':
            similarity = torch.mm(features, features.t())  # (B, B)

        elif self.distance == 'l2':
            diff = features.unsqueeze(1) - features.unsqueeze(0)  # (B, B, C)
            dist = torch.norm(diff, p=2, dim=2)  # (B, B)
            similarity = torch.exp(-dist / self.temperature)
 
        elif self.distance == 'kl':
            features_prob = F.softmax(features / self.temperature, dim=1)
            features_prob = features_prob + 1e-8  # Avoid log(0)
            log_p = features_prob.log()

            kl_matrix = []
            for i in range(B):
                kl_row = (features_prob * (log_p[i:i+1] - log_p)).sum(dim=1)
                kl_matrix.append(kl_row)

            kl_matrix = torch.stack(kl_matrix, dim=0)  # (B, B)
            similarity = torch.exp(-kl_matrix)

        else:
            raise ValueError(f"Unknown distance: {self.distance}")

        return similarity

    def forward(self, x, T, H, W, return_features=False):

        x = rearrange(x, "b c (t h w) -> (b t) c h w", t=T, h=H, w=W)
        features = self.compute_channel_features(x)  # (B, C)

        similarity = self.compute_pairwise_similarity(features)  # (B, B)

        B = similarity.shape[0]

        avg_similarity = similarity.sum() / (B * B)

        loss = 1.0 - avg_similarity

        if return_features:
            return loss, features
        else:
            return loss

