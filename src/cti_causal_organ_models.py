"""Model architectures for Causal Skill Organ admission test.

Protocol: CSO_ADMISSION_V1 (locked Jul 26 2026).

Donor: ~19M recurrent-state Transformer with designated latent state slot.
Host T: ~0.9M Transformer (same block family, smaller). <=10% donor inference compute.
Host G: ~1.3M GRU.

The donor processes one instruction at a time through a recurrent state.
The state slot is an architectural causal boundary -- true register values
are NEVER provided there.
"""
from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from cti_causal_register_transducer import MOD, NUM_OPS, NUM_REGS, SOCKET_SEED

NUM_INPUT_TOKENS = MOD + NUM_OPS
REG_OFFSET = 0
OP_OFFSET = MOD

MAX_ORGAN_D_STATE = 32
MAX_ORGAN_PARAMS = 32_000
MAX_ORGAN_BYTES = 65_536


class RecurrentTransformerDonor(nn.Module):
    """~19M recurrent-state Transformer donor.

    Processes a sequence of instructions one at a time with an explicit
    recurrent state vector. LayerNorm stabilizes the residual recurrence.
    """

    def __init__(
        self,
        d_model: int = 384,
        n_layers: int = 10,
        n_heads: int = 6,
        d_state: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state

        self.reg_embed = nn.Embedding(MOD, d_model)
        self.op_embed = nn.Embedding(NUM_OPS, d_model)

        self.state_init = nn.Sequential(
            nn.Linear(NUM_REGS * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_state),
        )

        self.state_to_input = nn.Linear(d_state, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.state_update = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_state),
        )

        self.state_norm = nn.LayerNorm(d_state)

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_state, d_model),
                nn.GELU(),
                nn.Linear(d_model, MOD),
            )
            for _ in range(NUM_REGS)
        ])

    def init_state(self, init_regs: torch.Tensor) -> torch.Tensor:
        """Initialize recurrent state from register values.
        init_regs: (B, 4) long tensor of register values in [0, 15].
        Returns: (B, d_state) state vector.
        """
        reg_emb = self.reg_embed(init_regs)
        reg_flat = reg_emb.reshape(reg_emb.size(0), -1)
        return self.state_init(reg_flat)

    def step(self, state: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        """Process one instruction step.
        state: (B, d_state)
        op: (B,) long tensor of operation indices in [0, 7].
        Returns: (B, d_state) updated state.
        """
        state_input = self.state_to_input(state)
        op_input = self.op_embed(op)

        x = torch.stack([state_input, op_input], dim=1)
        x = self.transformer(x)

        new_state = self.state_update(x[:, 0, :])
        return self.state_norm(state + new_state)

    def forward(
        self,
        init_regs: torch.Tensor,
        ops: torch.Tensor,
        ops_mask: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass.
        init_regs: (B, 4) initial register values.
        ops: (B, L) operation sequence.
        ops_mask: (B, L) bool mask (True = valid step).
        Returns: (logits, final_state)
        """
        state = self.init_state(init_regs)
        B, L = ops.shape

        for t in range(L):
            if ops_mask is not None:
                mask_t = ops_mask[:, t]
                step_state = self.step(state, ops[:, t])
                state = torch.where(mask_t.unsqueeze(1), step_state, state)
            else:
                state = self.step(state, ops[:, t])

        logits = tuple(head(state) for head in self.output_heads)
        return logits, state

    def get_state_at_step(
        self,
        init_regs: torch.Tensor,
        ops: torch.Tensor,
        step: int,
    ) -> torch.Tensor:
        """Get the recurrent state after a specific step (for intervention)."""
        state = self.init_state(init_regs)
        for t in range(min(step, ops.shape[1])):
            state = self.step(state, ops[:, t])
        return state


def _make_frozen_socket(d_in: int, d_out: int, seed: int = SOCKET_SEED) -> nn.Linear:
    """Create a parameter-free random projection socket.
    Fixed initialization, frozen (no gradients). Task-independent.
    Saves/restores global RNG and pins num_threads=1 so orthogonal_
    init is deterministic regardless of host thread configuration.
    """
    rng_state = torch.random.get_rng_state()
    saved_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    socket = nn.Linear(d_in, d_out, bias=False)
    nn.init.orthogonal_(socket.weight, gain=1.0)
    torch.set_num_threads(saved_threads)
    torch.random.set_rng_state(rng_state)
    socket.weight.requires_grad_(False)
    return socket


class TransformerHost(nn.Module):
    """~0.9M Transformer host. <=10% donor inference compute.

    Socket is a frozen random projection (task-independent).
    Consumes organ.read() messages, not raw organ state.
    """

    def __init__(
        self,
        d_model: int = 128,
        n_layers: int = 4,
        n_heads: int = 4,
        d_state: int = 64,
        d_organ: int = MAX_ORGAN_D_STATE,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_organ = d_organ

        self.reg_embed = nn.Embedding(MOD, d_model)
        self.op_embed = nn.Embedding(NUM_OPS, d_model)

        self.state_init = nn.Sequential(
            nn.Linear(NUM_REGS * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_state),
        )

        self.state_to_input = nn.Linear(d_state, d_model)
        self.state_norm = nn.LayerNorm(d_state)

        self.organ_socket = _make_frozen_socket(d_organ, d_model)

        layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(layer, num_layers=n_layers)

        self.state_update = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_state),
        )

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_state, d_model),
                nn.GELU(),
                nn.Linear(d_model, MOD),
            )
            for _ in range(NUM_REGS)
        ])

    def init_state(self, init_regs: torch.Tensor) -> torch.Tensor:
        reg_emb = self.reg_embed(init_regs)
        reg_flat = reg_emb.reshape(reg_emb.size(0), -1)
        return self.state_init(reg_flat)

    def step(
        self,
        state: torch.Tensor,
        op: torch.Tensor,
        organ_msg: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        state_input = self.state_to_input(state)
        op_input = self.op_embed(op)

        if organ_msg is not None:
            organ_input = self.organ_socket(organ_msg)
            x = torch.stack([state_input, op_input, organ_input], dim=1)
        else:
            x = torch.stack([state_input, op_input], dim=1)

        x = self.transformer(x)
        new_state = self.state_update(x[:, 0, :])
        return self.state_norm(state + new_state)

    def forward(
        self,
        init_regs: torch.Tensor,
        ops: torch.Tensor,
        organ=None,
        ops_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        state = self.init_state(init_regs)
        B, L = ops.shape

        if ops_mask is not None and L > 1:
            reactivated = (~ops_mask[:, :-1]) & ops_mask[:, 1:]
            assert not reactivated.any(), (
                "ops_mask must be contiguous right-padding (no holes)"
            )

        organ_state = None
        if organ is not None:
            organ_state = organ.init_state(init_regs)

        for t in range(L):
            active = ops_mask[:, t] if ops_mask is not None else None

            organ_msg = None
            if organ is not None:
                new_organ = organ.step(organ_state, ops[:, t])
                if active is not None:
                    organ_state = torch.where(
                        active.unsqueeze(1), new_organ, organ_state
                    )
                else:
                    organ_state = new_organ
                organ_msg = organ.read(organ_state)

            step_state = self.step(state, ops[:, t], organ_msg)
            if active is not None:
                state = torch.where(active.unsqueeze(1), step_state, state)
            else:
                state = step_state

        logits = tuple(head(state) for head in self.output_heads)
        return logits, state


class GRUHost(nn.Module):
    """~1.3M GRU host.

    Socket is a frozen random projection (task-independent).
    Consumes organ.read() messages, not raw organ state.
    """

    def __init__(
        self,
        d_model: int = 256,
        n_layers: int = 3,
        d_state: int = 192,
        d_organ: int = MAX_ORGAN_D_STATE,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_organ = d_organ
        self.n_layers = n_layers

        self.reg_embed = nn.Embedding(MOD, d_model)
        self.op_embed = nn.Embedding(NUM_OPS, d_model)

        self.state_init = nn.Sequential(
            nn.Linear(NUM_REGS * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, n_layers * d_state),
        )

        self.organ_socket = _make_frozen_socket(d_organ, d_model)

        self.gru = nn.GRU(
            input_size=d_model,
            hidden_size=d_state,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        self.output_heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_state, d_model),
                nn.GELU(),
                nn.Linear(d_model, MOD),
            )
            for _ in range(NUM_REGS)
        ])

    def init_hidden(self, init_regs: torch.Tensor) -> torch.Tensor:
        reg_emb = self.reg_embed(init_regs)
        reg_flat = reg_emb.reshape(reg_emb.size(0), -1)
        h = self.state_init(reg_flat)
        h = h.reshape(-1, self.n_layers, self.d_state)
        h = h.permute(1, 0, 2).contiguous()
        return h

    def forward(
        self,
        init_regs: torch.Tensor,
        ops: torch.Tensor,
        organ=None,
        ops_mask: Optional[torch.Tensor] = None,
    ) -> tuple:
        B, L = ops.shape

        if ops_mask is not None and L > 1:
            reactivated = (~ops_mask[:, :-1]) & ops_mask[:, 1:]
            assert not reactivated.any(), (
                "ops_mask must be contiguous right-padding (no holes)"
            )

        hidden = self.init_hidden(init_regs)

        op_emb = self.op_embed(ops)

        if organ is not None:
            organ_state = organ.init_state(init_regs)
            organ_inputs = []
            for t in range(L):
                new_organ = organ.step(organ_state, ops[:, t])
                if ops_mask is not None:
                    active = ops_mask[:, t]
                    organ_state = torch.where(
                        active.unsqueeze(1), new_organ, organ_state
                    )
                else:
                    organ_state = new_organ
                organ_msg = organ.read(organ_state)
                organ_inputs.append(self.organ_socket(organ_msg))
            organ_seq = torch.stack(organ_inputs, dim=1)
            gru_input = op_emb + organ_seq
        else:
            gru_input = op_emb

        output, hidden = self.gru(gru_input, hidden)

        if ops_mask is not None:
            lengths = ops_mask.long().sum(dim=1)
            last_idx = (lengths - 1).clamp(min=0)
            final = output[torch.arange(B, device=output.device), last_idx]
        else:
            final = output[:, -1, :]

        logits = tuple(head(final) for head in self.output_heads)
        return logits, final


class CausalOrgan(nn.Module):
    """Causal Skill Organ: compact executable state-transition module.

    Maximum 32-dim state, 32K quantized parameters, 64 KiB.
    Frozen after extraction, identical bytes for both hosts.

    z_{t+1} = F(z_t, U_t)
    m_t = G(z_t)
    """

    def __init__(self, d_state: int = 32, d_hidden: int = 48):
        super().__init__()
        assert d_state <= MAX_ORGAN_D_STATE, (
            f"Organ d_state={d_state} > max {MAX_ORGAN_D_STATE}"
        )
        self.d_state = d_state

        self.reg_init = nn.Sequential(
            nn.Linear(NUM_REGS * 16, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_state),
        )

        self.op_embed = nn.Embedding(NUM_OPS, d_hidden)

        self.transition = nn.Sequential(
            nn.Linear(d_state + d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, d_state),
        )

        self.state_norm = nn.LayerNorm(d_state)
        self.readout = nn.Linear(d_state, d_state)

        n_params = sum(p.numel() for p in self.parameters())
        assert n_params <= MAX_ORGAN_PARAMS, (
            f"Organ has {n_params} params > max {MAX_ORGAN_PARAMS}"
        )

        import io
        buf = io.BytesIO()
        torch.save(self.state_dict(), buf)
        serialized = buf.tell()
        assert serialized <= MAX_ORGAN_BYTES, (
            f"Organ serialized size {serialized:,} bytes > "
            f"{MAX_ORGAN_BYTES:,} byte limit"
        )

    def init_state(self, init_regs: torch.Tensor) -> torch.Tensor:
        """Initialize organ state from register values (one-hot encoded)."""
        B = init_regs.size(0)
        one_hot = F.one_hot(init_regs.long(), MOD).float()
        flat = one_hot.reshape(B, -1)
        return self.reg_init(flat)

    def step(self, state: torch.Tensor, op: torch.Tensor) -> torch.Tensor:
        """One transition step. Returns updated state."""
        op_emb = self.op_embed(op)
        x = torch.cat([state, op_emb], dim=1)
        return self.state_norm(state + self.transition(x))

    def read(self, state: torch.Tensor) -> torch.Tensor:
        """Readout message from current state."""
        return self.readout(state)

    def forward(
        self,
        init_regs: torch.Tensor,
        ops: torch.Tensor,
    ) -> torch.Tensor:
        """Full rollout. Returns final organ state."""
        state = self.init_state(init_regs)
        for t in range(ops.shape[1]):
            state = self.step(state, ops[:, t])
        return state


# ---------------------------------------------------------------------------
# FLOP estimation
# ---------------------------------------------------------------------------

def estimate_step_macs(model: nn.Module) -> int:
    """Per-step MAC estimate for compute budget verification.
    Attention: QKV projections + output proj = 4*d*d*seq, scores = 2*seq*seq*d.
    FFN: up + down projections = 2 * d * 4d * seq = 8*d*d*seq.
    """
    if isinstance(model, RecurrentTransformerDonor):
        d = model.d_model
        n = model.transformer.num_layers
        seq = 2
        attn = 4 * d * d * seq + 2 * seq * seq * d
        ffn = 8 * d * d * seq
        per_layer = attn + ffn
        return per_layer * n
    elif isinstance(model, TransformerHost):
        d = model.d_model
        n = model.transformer.num_layers
        seq = 3
        attn = 4 * d * d * seq + 2 * seq * seq * d
        ffn = 8 * d * d * seq
        per_layer = attn + ffn
        return per_layer * n
    elif isinstance(model, GRUHost):
        d = model.d_model
        h = model.d_state
        n = model.n_layers
        return n * 3 * (d * h + h * h)
    return 0


def verify_compute_budget(donor, host_t, threshold: float = 0.10):
    """Assert host + organ inference compute <= threshold * donor."""
    donor_macs = estimate_step_macs(donor)
    host_macs = estimate_step_macs(host_t)
    ratio = host_macs / donor_macs if donor_macs > 0 else 0
    print(f"  Donor per-step MACs: ~{donor_macs:,}")
    print(f"  Host T per-step MACs: ~{host_macs:,}")
    print(f"  Ratio: {ratio:.1%} (limit: {threshold:.0%})")
    assert ratio <= threshold, (
        f"Host T compute ratio {ratio:.1%} > {threshold:.0%} limit"
    )
    return ratio


# ---------------------------------------------------------------------------
# Factory functions
# ---------------------------------------------------------------------------

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_donor(**kwargs) -> RecurrentTransformerDonor:
    model = RecurrentTransformerDonor(**kwargs)
    n = count_parameters(model)
    n_train = count_trainable_parameters(model)
    print(f"Donor: {n:,} parameters ({n/1e6:.1f}M), {n_train:,} trainable")
    return model


def create_host_transformer(**kwargs) -> TransformerHost:
    model = TransformerHost(**kwargs)
    n = count_parameters(model)
    n_train = count_trainable_parameters(model)
    print(f"Host T: {n:,} parameters ({n/1e6:.1f}M), "
          f"{n_train:,} trainable (socket frozen)")
    return model


def create_host_gru(**kwargs) -> GRUHost:
    model = GRUHost(**kwargs)
    n = count_parameters(model)
    n_train = count_trainable_parameters(model)
    print(f"Host G: {n:,} parameters ({n/1e6:.1f}M), "
          f"{n_train:,} trainable (socket frozen)")
    return model


def measure_organ_serialized_bytes(organ: CausalOrgan) -> int:
    """Measure actual serialized state_dict size in bytes."""
    import io
    buf = io.BytesIO()
    torch.save(organ.state_dict(), buf)
    return buf.tell()


def create_organ(**kwargs) -> CausalOrgan:
    model = CausalOrgan(**kwargs)
    n = count_parameters(model)
    actual_bytes = measure_organ_serialized_bytes(model)
    print(f"Organ: {n:,} parameters, {actual_bytes:,} serialized bytes "
          f"({actual_bytes/1024:.1f} KiB, limit: {MAX_ORGAN_BYTES/1024:.0f} KiB)")
    assert actual_bytes <= MAX_ORGAN_BYTES, (
        f"Organ serialized size {actual_bytes:,} bytes > "
        f"{MAX_ORGAN_BYTES:,} byte limit"
    )
    return model


if __name__ == "__main__":
    print("=" * 60)
    print("CAUSAL SKILL ORGAN: MODEL PARAMETER CENSUS")
    print("=" * 60)
    print()

    donor = create_donor()
    print()
    host_t = create_host_transformer()
    print()
    host_g = create_host_gru()
    print()
    organ = create_organ()
    print()

    print("--- Compute budget check ---")
    verify_compute_budget(donor, host_t)
    print()

    print("--- Smoke test (batch=4, L=8) ---")
    B, L = 4, 8
    init_regs = torch.randint(0, MOD, (B, NUM_REGS))
    ops = torch.randint(0, NUM_OPS, (B, L))

    print("Donor forward...")
    logits_d, state_d = donor(init_regs, ops)
    print(f"  Logits: {[l.shape for l in logits_d]}, State: {state_d.shape}")
    assert all(torch.isfinite(l).all() for l in logits_d), "Donor logits not finite"
    assert torch.isfinite(state_d).all(), "Donor state not finite"

    print("Donor forward (L=32 stability check)...")
    ops_long = torch.randint(0, NUM_OPS, (B, 32))
    logits_long, state_long = donor(init_regs, ops_long)
    assert torch.isfinite(state_long).all(), "Donor state not finite at L=32"
    state_rms = state_long.float().pow(2).mean().sqrt().item()
    print(f"  State RMS at L=32: {state_rms:.2f}")

    print("Host T forward (no organ)...")
    logits_t, state_t = host_t(init_regs, ops)
    print(f"  Logits: {[l.shape for l in logits_t]}, State: {state_t.shape}")

    print("Host G forward (no organ)...")
    logits_g, state_g = host_g(init_regs, ops)
    print(f"  Logits: {[l.shape for l in logits_g]}, State: {state_g.shape}")

    print("Organ forward...")
    organ_out = organ(init_regs, ops)
    print(f"  Organ state: {organ_out.shape}")

    print("Host T forward (with organ)...")
    logits_to, state_to = host_t(init_regs, ops, organ=organ)
    print(f"  Logits: {[l.shape for l in logits_to]}, State: {state_to.shape}")

    print("Host G forward (with organ)...")
    logits_go, state_go = host_g(init_regs, ops, organ=organ)
    print(f"  Logits: {[l.shape for l in logits_go]}, State: {state_go.shape}")

    print()

    n_donor = count_parameters(donor)
    n_host_t = count_parameters(host_t)
    n_host_g = count_parameters(host_g)
    n_organ = count_parameters(organ)

    print(f"Compression ratio (donor/host_t): {n_donor/n_host_t:.1f}x")
    print(f"Compression ratio (donor/host_g): {n_donor/n_host_g:.1f}x")
    print(f"Organ as % of donor: {100*n_organ/n_donor:.2f}%")
    print(f"Organ size: {n_organ*4/1024:.1f} KiB (limit: 64 KiB)")

    print()
    print("ALL SMOKE TESTS PASSED")
