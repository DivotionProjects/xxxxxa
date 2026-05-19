from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class AttentionLayoutMetadata:
    item_lengths: tuple[int, ...]
    adjacency: torch.Tensor
    block_size: int = 64

    def to(self, device: torch.device | str | None = None) -> AttentionLayoutMetadata:
        # The layout is tiny and only needed to build CPU-side masks. Keeping it
        # on CPU avoids accidental GPU synchronization in dataloader transfers.
        return self


@dataclass
class _KVCachePlan:
    query_indices: torch.Tensor
    key_indices: torch.Tensor
    current_cu_seqlens: torch.Tensor
    segment_new_lengths: tuple[int, ...]
    is_incremental: bool


@dataclass
class CacheForwardInputs:
    tokens: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: Any


class KVCache:
    def __init__(self, pad_token_ids: int = 128274):
        self.k: dict[int, torch.Tensor] = {}
        self.v: dict[int, torch.Tensor] = {}
        self.cu_seqlens: dict[int, torch.Tensor] = {}

        self.pad_token_ids = pad_token_ids
        self.padding_index: int | None = None

        self._plan: _KVCachePlan | None = None

    def has_cache(self, cache_index: int = 0) -> bool:
        return cache_index in self.k and cache_index in self.v and cache_index in self.cu_seqlens

    def check_padding(
        self,
        input_tokens: torch.Tensor,
        cu_seqlens: torch.Tensor,
    ) -> None:
        if self.padding_index is not None:
            return
        for idx, (start, end) in enumerate(self._iter_segments(cu_seqlens)):
            if self._is_padding_segment(input_tokens[start:end]):
                self.padding_index = idx
                return

    def prepare_text_inputs(
        self,
        input_tokens: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        attention_mask: Any,
        attention_layout: AttentionLayoutMetadata | None,
    ) -> CacheForwardInputs:
        plan = self._build_plan(input_tokens, cu_seqlens)
        self._plan = plan

        if not plan.is_incremental:
            return CacheForwardInputs(
                tokens=input_tokens,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        if plan.query_indices.numel() == 0:
            raise ValueError("KV-cache incremental step has no uncached non-padding tokens.")
        if attention_layout is None:
            raise ValueError("KV-cache incremental text generation requires attention_layout metadata.")

        return CacheForwardInputs(
            tokens=input_tokens.index_select(0, plan.query_indices.to(input_tokens.device)),
            position_ids=position_ids.index_select(0, plan.query_indices.to(position_ids.device)),
            attention_mask=(
                None if attention_mask is None else self.build_block_mask(attention_layout, position_ids.device)
            ),
        )

    def cut_input_tokens(
        self,
        input_tokens: torch.Tensor,
        cu_seqlens: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self._plan is None or not self._plan.is_incremental:
            return input_tokens
        return input_tokens.index_select(0, self._plan.query_indices.to(input_tokens.device))

    def cut_field_offsets(
        self,
        field_offsets: dict[str, tuple[int, int]],
    ) -> dict[str, tuple[int, int]]:
        if self._plan is None or not self._plan.is_incremental:
            return dict(field_offsets)

        rebased: dict[str, tuple[int, int]] = {}
        query_indices = self._plan.query_indices.detach().cpu()
        for key, (start, end) in field_offsets.items():
            selected = ((query_indices >= start) & (query_indices < end)).nonzero(as_tuple=False).flatten()
            if selected.numel() == 0:
                rebased[key] = (0, 0)
            else:
                rebased[key] = (int(selected[0].item()), int(selected[-1].item()) + 1)
        return rebased

    def cut_modality_metadata(
        self,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        field_offsets: dict[str, tuple[int, int]],
        field_name: str,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self._plan is None or not self._plan.is_incremental:
            return position_ids, cu_seqlens

        start, end = field_offsets[field_name]
        local_indices = self._selected_local_indices(start, end).to(position_ids.device)
        new_position_ids = position_ids.index_select(0, local_indices)

        counts: list[int] = []
        local_indices_cpu = local_indices.detach().cpu()
        for seq_start, seq_end in self._iter_segments(cu_seqlens):
            count = int(((local_indices_cpu >= seq_start) & (local_indices_cpu < seq_end)).sum().item())
            counts.append(count)

        new_cu_seqlens = torch.tensor(
            [0, *np.cumsum(counts).tolist()],
            dtype=cu_seqlens.dtype,
            device=cu_seqlens.device,
        )
        return new_position_ids, new_cu_seqlens

    def update(
        self,
        key_states: torch.Tensor,
        value_states: torch.Tensor,
        cu_seqlens: torch.Tensor | None,
        cache_index: int = -1,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not use_cache:
            return key_states, value_states
        if self._plan is None:
            raise RuntimeError("KVCache.prepare_text_inputs must be called before KVCache.update.")

        plan = self._plan
        current_cu = plan.current_cu_seqlens.detach().cpu()

        if not plan.is_incremental:
            key_keep_indices = plan.key_indices.to(key_states.device)
            stored_key_states = key_states.index_select(0, key_keep_indices)
            stored_value_states = value_states.index_select(0, key_keep_indices.to(value_states.device))

            self.k[cache_index] = stored_key_states.clone().cpu().detach()
            self.v[cache_index] = stored_value_states.clone().cpu().detach()
            self.cu_seqlens[cache_index] = current_cu
            return key_states, value_states

        if cache_index not in self.k or cache_index not in self.v or cache_index not in self.cu_seqlens:
            raise RuntimeError(f"Missing KV-cache state for layer {cache_index}.")

        past_key_states = self.k[cache_index].to(key_states.device)
        past_value_states = self.v[cache_index].to(value_states.device)
        cached_cu = self.cu_seqlens[cache_index].to(key_states.device)

        key_parts: list[torch.Tensor] = []
        value_parts: list[torch.Tensor] = []
        new_start = 0
        for segment_idx, new_len in enumerate(plan.segment_new_lengths):
            cache_start = int(cached_cu[segment_idx].item())
            cache_end = int(cached_cu[segment_idx + 1].item())
            if cache_end > cache_start:
                key_parts.append(past_key_states[cache_start:cache_end])
                value_parts.append(past_value_states[cache_start:cache_end])

            if new_len > 0:
                new_end = new_start + new_len
                key_parts.append(key_states[new_start:new_end])
                value_parts.append(value_states[new_start:new_end])
                new_start = new_end

        if new_start != key_states.shape[0]:
            raise RuntimeError("KV-cache plan does not match projected key/value length.")

        next_key_states = torch.cat(key_parts, dim=0) if key_parts else key_states[:0]
        next_value_states = torch.cat(value_parts, dim=0) if value_parts else value_states[:0]

        self.k[cache_index] = next_key_states.clone().cpu().detach()
        self.v[cache_index] = next_value_states.clone().cpu().detach()
        self.cu_seqlens[cache_index] = current_cu

        return next_key_states, next_value_states

    def selected_dense_attention_mask(self, attention_layout: AttentionLayoutMetadata) -> torch.Tensor:
        if self._plan is None:
            raise RuntimeError("KVCache.prepare_text_inputs must be called before building the cache mask.")
        dense_mask = self._build_dense_attention_mask(attention_layout)
        q_indices = self._plan.query_indices.detach().cpu()
        k_indices = self._plan.key_indices.detach().cpu()
        return dense_mask.index_select(0, q_indices).index_select(1, k_indices)

    def build_block_mask(self, attention_layout: AttentionLayoutMetadata, device: torch.device | str) -> Any:
        selected_mask = self.selected_dense_attention_mask(attention_layout)
        block_table, mask_table = self._pack_dense_mask(selected_mask, attention_layout.block_size)

        from omni_attention import BlockMask

        return BlockMask(block_table, mask_table, len(mask_table) - 1).to(device)

    def _build_plan(self, input_tokens: torch.Tensor, cu_seqlens: torch.Tensor) -> _KVCachePlan:
        current_lengths: list[int] = []
        key_indices: list[torch.Tensor] = []

        for idx, (start, end) in enumerate(self._iter_segments(cu_seqlens)):
            segment = input_tokens[start:end]
            if self._is_padding_segment(segment):
                if self.padding_index is None:
                    self.padding_index = idx
                current_lengths.append(0)
                continue

            current_lengths.append(end - start)
            key_indices.append(torch.arange(start, end, dtype=torch.long, device=input_tokens.device))

        if self.has_cache(0):
            cached_lengths = torch.diff(self.cu_seqlens[0]).tolist()
            if len(cached_lengths) != len(current_lengths):
                raise ValueError("KV-cache segment layout changed between decoding steps.")
        else:
            cached_lengths = [0] * len(current_lengths)

        query_indices: list[torch.Tensor] = []
        new_lengths: list[int] = []
        for cached_len, current_len, (start, end) in zip(
            cached_lengths,
            current_lengths,
            self._iter_segments(cu_seqlens),
            strict=True,
        ):
            if current_len < cached_len:
                raise ValueError("KV-cache only supports append-only non-padding text segments.")

            new_len = current_len - cached_len
            new_lengths.append(new_len)
            if new_len > 0:
                query_indices.append(
                    torch.arange(start + cached_len, end, dtype=torch.long, device=input_tokens.device)
                )

        compact_cu = torch.tensor(
            [0, *np.cumsum(current_lengths).tolist()],
            dtype=torch.int32,
        )
        all_key_indices = (
            torch.cat(key_indices, dim=0)
            if key_indices
            else torch.empty(0, dtype=torch.long, device=input_tokens.device)
        )

        if self.has_cache(0):
            all_query_indices = (
                torch.cat(query_indices, dim=0)
                if query_indices
                else torch.empty(0, dtype=torch.long, device=input_tokens.device)
            )
            return _KVCachePlan(
                query_indices=all_query_indices,
                key_indices=all_key_indices,
                current_cu_seqlens=compact_cu,
                segment_new_lengths=tuple(new_lengths),
                is_incremental=True,
            )

        full_indices = torch.arange(input_tokens.shape[0], dtype=torch.long, device=input_tokens.device)
        return _KVCachePlan(
            query_indices=full_indices,
            key_indices=all_key_indices,
            current_cu_seqlens=compact_cu,
            segment_new_lengths=tuple(current_lengths),
            is_incremental=False,
        )

    def _selected_local_indices(self, start: int, end: int) -> torch.Tensor:
        if self._plan is None:
            return torch.arange(end - start, dtype=torch.long)
        query_indices = self._plan.query_indices.detach().cpu()
        selected = query_indices[(query_indices >= start) & (query_indices < end)] - start
        return selected.to(torch.long)

    def _build_dense_attention_mask(self, attention_layout: AttentionLayoutMetadata) -> torch.Tensor:
        item_lengths = attention_layout.item_lengths
        offsets = np.cumsum([0, *item_lengths]).tolist()
        total_len = offsets[-1]
        dense_mask = torch.zeros((total_len, total_len), dtype=torch.bool)
        adjacency = attention_layout.adjacency.detach().cpu().to(torch.int64)

        for row_idx in range(len(item_lengths)):
            q_start, q_end = offsets[row_idx], offsets[row_idx + 1]
            q_len = q_end - q_start
            if q_len == 0:
                continue
            for col_idx in range(len(item_lengths)):
                mask_type = int(adjacency[row_idx, col_idx].item())
                if mask_type == 0:
                    continue

                k_start, k_end = offsets[col_idx], offsets[col_idx + 1]
                k_len = k_end - k_start
                if k_len == 0:
                    continue

                if mask_type == 1:
                    q_pos = torch.arange(q_len).unsqueeze(1)
                    k_pos = torch.arange(k_len).unsqueeze(0)
                    dense_mask[q_start:q_end, k_start:k_end] = k_pos <= q_pos
                elif mask_type == 2:
                    dense_mask[q_start:q_end, k_start:k_end] = True
                else:
                    raise ValueError(f"Unsupported attention mask type {mask_type}.")

        return dense_mask

    @staticmethod
    def _pack_dense_mask(dense_mask: torch.Tensor, block_size: int) -> tuple[np.ndarray, np.ndarray]:
        q_len, k_len = dense_mask.shape
        q_blocks = max((q_len + block_size - 1) // block_size, 1)
        k_blocks = max((k_len + block_size - 1) // block_size, 1)

        block_table = np.zeros((1, q_blocks, k_blocks), dtype=np.int32)
        mask_table: list[np.ndarray] = [
            np.zeros((block_size, block_size), dtype=np.uint8),
            np.ones((block_size, block_size), dtype=np.uint8),
        ]
        partial_index: dict[bytes, int] = {}

        for q_block in range(q_blocks):
            q_start = q_block * block_size
            q_end = min(q_start + block_size, q_len)
            for k_block in range(k_blocks):
                k_start = k_block * block_size
                k_end = min(k_start + block_size, k_len)
                submask = dense_mask[q_start:q_end, k_start:k_end]
                if submask.numel() == 0 or not bool(submask.any().item()):
                    continue
                if bool(submask.all().item()):
                    block_table[0, q_block, k_block] = 1
                    continue

                packed = np.zeros((block_size, block_size), dtype=np.uint8)
                packed[: q_end - q_start, : k_end - k_start] = submask.numpy().astype(np.uint8)
                key = packed.tobytes()
                mask_idx = partial_index.get(key)
                if mask_idx is None:
                    mask_idx = len(mask_table)
                    partial_index[key] = mask_idx
                    mask_table.append(packed)
                block_table[0, q_block, k_block] = mask_idx

        return block_table, np.stack(mask_table, axis=0)

    def _is_padding_segment(self, segment: torch.Tensor) -> bool:
        return bool(segment.numel() > 0 and torch.all(segment == self.pad_token_ids).item())

    @staticmethod
    def _iter_segments(cu_seqlens: torch.Tensor) -> list[tuple[int, int]]:
        cu = cu_seqlens.detach().cpu().tolist()
        return [(int(cu[idx]), int(cu[idx + 1])) for idx in range(len(cu) - 1)]
