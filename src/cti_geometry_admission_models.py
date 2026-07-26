"""Geometry Admission Test: model architectures.

Teacher Transformer (19.5M), Student Transformer (1.9M), GRU Student (1.8M).
All per the Stage A specification.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cti_geometry_admission_automaton import VOCAB_SIZE, PAD_TOKEN, NUM_STATES


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class CausalSelfAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int, max_positions: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out_proj = nn.Linear(dim, dim, bias=False)
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_positions, max_positions, dtype=torch.bool)),
        )

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        scale = 1.0 / math.sqrt(self.head_dim)
        attn = (q @ k.transpose(-2, -1)) * scale

        causal = self.causal_mask[:T, :T]
        attn = attn.masked_fill(~causal.unsqueeze(0).unsqueeze(0), float("-inf"))

        if attention_mask is not None:
            key_mask = attention_mask.unsqueeze(1).unsqueeze(2)
            attn = attn.masked_fill(key_mask == 0, float("-inf"))

        attn = F.softmax(attn, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(B, T, C)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ff_dim: int, max_positions: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = CausalSelfAttention(dim, n_heads, max_positions)
        self.norm2 = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ff_dim)

    def forward(self, x: torch.Tensor, attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        x = x + self.attn(self.norm1(x), attention_mask)
        x = x + self.ffn(self.norm2(x))
        return x


class AutomatonTransformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        dim: int,
        n_heads: int,
        ff_dim: int,
        max_positions: int = 65,
    ):
        super().__init__()
        self.dim = dim
        self.token_emb = nn.Embedding(VOCAB_SIZE, dim)
        self.pos_emb = nn.Embedding(max_positions, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim, n_heads, ff_dim, max_positions)
            for _ in range(n_layers)
        ])
        self.norm = RMSNorm(dim)
        self.classifier = nn.Linear(dim, NUM_STATES, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> dict:
        B, T = input_ids.shape
        positions = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        hidden_states = [x] if return_hidden_states else None

        for block in self.blocks:
            x = block(x, attention_mask)
            if return_hidden_states:
                hidden_states.append(x)

        x = self.norm(x)

        last_token_idx = attention_mask.sum(dim=1) - 1
        last_hidden = x[torch.arange(B, device=x.device), last_token_idx]
        logits = self.classifier(last_hidden)

        result = {"logits": logits}
        if return_hidden_states:
            result["hidden_states"] = hidden_states
        return result


class AutomatonGRU(nn.Module):
    def __init__(
        self,
        embed_dim: int = 128,
        hidden_dim: int = 224,
        n_layers: int = 6,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        self.token_emb = nn.Embedding(VOCAB_SIZE, embed_dim)
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        self.gru_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(n_layers):
            self.gru_layers.append(nn.GRU(hidden_dim, hidden_dim, batch_first=True))
            self.norms.append(RMSNorm(hidden_dim))

        self.classifier = nn.Linear(hidden_dim, NUM_STATES, bias=True)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        return_hidden_states: bool = False,
    ) -> dict:
        B, T = input_ids.shape
        x = self.token_emb(input_ids)
        x = self.input_proj(x)

        hidden_states = [x] if return_hidden_states else None

        for gru, norm in zip(self.gru_layers, self.norms):
            x, _ = gru(x)
            x = norm(x)
            if return_hidden_states:
                hidden_states.append(x)

        last_token_idx = attention_mask.sum(dim=1) - 1
        last_hidden = x[torch.arange(B, device=x.device), last_token_idx]
        logits = self.classifier(last_hidden)

        result = {"logits": logits}
        if return_hidden_states:
            result["hidden_states"] = hidden_states
        return result


def create_teacher() -> AutomatonTransformer:
    return AutomatonTransformer(
        n_layers=12, dim=384, n_heads=6, ff_dim=896, max_positions=65,
    )


def create_transformer_student() -> AutomatonTransformer:
    return AutomatonTransformer(
        n_layers=6, dim=160, n_heads=5, ff_dim=448, max_positions=65,
    )


def create_gru_student() -> AutomatonGRU:
    return AutomatonGRU(embed_dim=128, hidden_dim=224, n_layers=6)


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


if __name__ == "__main__":
    teacher = create_teacher()
    t_student = create_transformer_student()
    g_student = create_gru_student()

    print(f"Teacher:             {count_parameters(teacher):>12,} params")
    print(f"Transformer student: {count_parameters(t_student):>12,} params")
    print(f"GRU student:         {count_parameters(g_student):>12,} params")
    print(f"Compression (T):     {count_parameters(teacher)/count_parameters(t_student):.2f}x")
    print(f"Compression (GRU):   {count_parameters(teacher)/count_parameters(g_student):.2f}x")

    batch = {
        "input_ids": torch.randint(0, 16, (4, 10)),
        "attention_mask": torch.ones(4, 10, dtype=torch.long),
    }

    for name, model in [("Teacher", teacher), ("T-Student", t_student), ("GRU-Student", g_student)]:
        out = model(batch["input_ids"], batch["attention_mask"], return_hidden_states=True)
        print(f"\n{name}:")
        print(f"  Logits: {out['logits'].shape}")
        print(f"  Hidden states: {len(out['hidden_states'])} checkpoints")
        for i, h in enumerate(out["hidden_states"]):
            print(f"    [{i}] {h.shape}")
