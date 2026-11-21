"""FSDP2 weight all-gathers on SDMA copy engines (XDIT_FSDP_SDMA=1: torch's patched SDMA ops; =pull: xfuser's
transport for a 2-rank FSDP group, with part of each shard relayed through another GPU, see PairAllGather).

FSDP2 (torch.distributed.fsdp.fully_shard) lets a module take a custom all-gather
(`FSDPModule.set_custom_all_gather`). Its contract, from torch's _fsdp_collectives.foreach_all_gather:
`allocate()` returns the gathered output on the copy-in stream, FSDP copies this rank's shard into its
slot of it (the gather is in place), then calls the comm on its all-gather stream and records an event
there that compute waits for before copying the weights out. The output is dropped after that copy-out.

Here the output stays ordinary memory (no lifetime to manage across ranks). The gather runs through
the SDMA all-gather of torch's patched ROCm SDMA ops (`_rocm_sdma._all_gather_and_consume`: every rank's
rows land in a peer-registered inbox, on the IPC workspace from core/distributed/ipc_workspace.py), and
each landed piece is copied from the inbox into FSDP's output on the all-gather stream. The engines move
the bytes between GPUs, so RCCL's copy kernels stop competing with the compute they overlap.
"""

from typing import Sequence

import torch
import torch.distributed as dist
from torch.distributed.fsdp._fully_shard._fsdp_api import AllGather

from xfuser.logger import init_logger

logger = init_logger(__name__)

_MAX_CHUNKS = 8  # pieces per pair all-gather (two parities' channels, and two sends each, within a call's limits)
_ROW_BYTES = 4096  # the shard is gathered as rows of this many bytes (padded to whole rows)
calls = {"sdma": 0}
# The calls of every module share one inbox (per group), but FSDP roots (each transformer) have their own
# all-gather streams: each call waits for the previous one's end, wherever it ran.
_last = {}


class SdmaAllGather(AllGather):
    """An FSDP2 all-gather over the patched torch ROCm SDMA transport (see the module docstring)."""

    def __init__(self, chunks: int = 0):
        self.chunks = chunks  # pieces per call (0: the transport's choice for the size)

    def allocate(self, size: Sequence[int], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.empty(size, dtype=dtype, device=device)

    def __call__(self, output_tensor, input_tensor, group: dist.ProcessGroup, async_op: bool = False):
        import torch.distributed._symmetric_memory._rocm_sdma as rs

        world = group.size()
        nbytes = input_tensor.numel() * input_tensor.element_size()
        rows = -(-nbytes // _ROW_BYTES)
        chunks = self.chunks or rs._chunks(rows, _ROW_BYTES)
        rows = -(-rows // chunks) * chunks  # whole pieces
        src = input_tensor.contiguous().view(torch.uint8).view(-1)
        if rows * _ROW_BYTES != nbytes:
            padded = torch.empty(rows * _ROW_BYTES, dtype=torch.uint8, device=src.device)
            padded[:nbytes].copy_(src)
            src = padded
        out = output_tensor.view(torch.uint8).view(world, nbytes)
        block = rows // chunks
        me = group.rank()
        same = out[me].data_ptr() == input_tensor.data_ptr()  # FSDP's in-place gather: own slot already filled

        def consume(c, pieces):
            # pieces[0]: (world, block, _ROW_BYTES), piece c of every rank's rows, landed in the inbox
            lo, hi = c * block * _ROW_BYTES, min((c + 1) * block * _ROW_BYTES, nbytes)
            if lo >= hi:
                return
            got = pieces[0].reshape(world, -1)
            for r in range(world):
                if r != me or not same:
                    out[r, lo:hi].copy_(got[r, : hi - lo])

        stream = torch.cuda.current_stream()
        prev = _last.get(group.group_name)
        if prev is not None:
            stream.wait_event(prev)
        rs._all_gather_and_consume([src.view(rows, _ROW_BYTES)], chunks, group.group_name, consume, False)
        _last[group.group_name] = stream.record_event()
        calls["sdma"] += 1
        return None


class _OnCompute(dist.Work):
    """FSDP2 calls a returned Work's wait() on the compute stream right before it copies the gathered weights out
    (foreach_all_gather_copy_out): the receive side of a pair all-gather runs there, serialized with compute."""

    def __init__(self, finish):
        super().__init__()
        self._finish = finish

    def wait(self, timeout=None) -> bool:
        if self._finish is not None:
            self._finish()
            self._finish = None
        return True

    def is_completed(self) -> bool:
        return self._finish is None


class PairAllGather(AllGather):
    """An FSDP2 all-gather for a 2-rank group over xfuser's SDMA pull relay (pull_relay.py, its own instance:
    tag "fsdp", its own engines). Nothing waits on FSDP's all-gather stream, and nothing spins next to compute (a
    resident wait wave makes a GEMM launched one work-group per CU run a second round):

    - Send (all-gather stream): the shard goes in pieces into the partner's inbox region (call n uses region
      n % R, R = transport._SLOTS), the share XDIT_FSDP_SDMA_RELAY_FRACTION (default 0.375; 0: all direct) through a
      buffer on a third GPU that the partner's engine pulls once released. Each piece's pushes wait in the engine
      for the stream (trigger) and for that (region, piece)'s credit.
    - Receive (compute stream, in the returned Work's wait(), which FSDP calls before its copy-out): wait for each
      piece's flags, copy it from the inbox into FSDP's output, and release the partner's credits for the region.

    The credits bound how far ahead a sender runs (R calls; FSDP's forward prefetch keeps 3 in flight), and this
    rank's slot of FSDP's output (the push source) is held for R + 2 calls, by when its pushes are done.
    """

    def allocate(self, size: Sequence[int], *, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
        return torch.empty(size, dtype=dtype, device=device)

    def __call__(self, output_tensor, input_tensor, group: dist.ProcessGroup, async_op: bool = False):
        import os

        from xfuser.core.distributed.sdma import pull_relay, transport

        fraction = float(os.environ.get("XDIT_FSDP_SDMA_RELAY_FRACTION", "0.375"))
        pair = pull_relay.pair_for(group, tag="fsdp", fraction=fraction)
        R = transport._SLOTS
        me = group.rank()
        nbytes = input_tensor.numel() * input_tensor.element_size()
        out = output_tensor.view(torch.uint8).view(2, nbytes)
        src = out[me] if out[me].data_ptr() == input_tensor.data_ptr() else input_tensor.contiguous().view(torch.uint8).view(-1)
        rb = -(-nbytes // 256) * 256
        inbox = pair.buffer("inbox", R * rb)
        chunks = max(1, min(int(os.environ.get("XDIT_FSDP_SDMA_CHUNKS", "4")), _MAX_CHUNKS, nbytes // (1 << 20) or 1))
        piece = -(-nbytes // chunks // 256) * 256
        bounds = [(c * piece, min((c + 1) * piece, nbytes)) for c in range(chunks) if c * piece < nbytes]
        region = pair.calls % R  # the call slot begin() takes next
        routes = [
            pair.route(
                ("fsdp_ag", nbytes, len(bounds), c, region),
                lambda lo=lo, hi=hi: [[(lo, inbox, region * rb + lo, hi - lo)]],
                region * _MAX_CHUNKS + c,
                credit=c,
                lane=("fsdp_ag", nbytes, len(bounds), c),
            )
            for c, (lo, hi) in enumerate(bounds)
        ]
        call = pair.begin()
        assert call.index == region
        stream = torch.cuda.current_stream()
        for route in routes:
            call.send(route, src, stream)
        call.end()
        pair.fsdp_held = (getattr(pair, "fsdp_held", []) + [output_tensor])[-(R + 2):]
        calls["pull"] = calls.get("pull", 0) + 1
        mine = inbox.local[region * rb : region * rb + nbytes]
        credits = [pair.credits(c) for c in range(len(bounds))]

        def finish():
            cur = torch.cuda.current_stream()
            for c, (lo, hi) in enumerate(bounds):
                call.wait(region * _MAX_CHUNKS + c, cur)
                out[1 - me, lo:hi].copy_(mine[lo:hi])
            # The region is free again: release the partner's credits for it (a store into its exported signals).
            for direct, relayed in credits:
                for ids, _, theirs in (direct, relayed):
                    transport.device_bytes(theirs[region], 8, out.device).view(torch.int64).zero_()

        return _OnCompute(finish)


def install(module: torch.nn.Module) -> bool:
    """Give an FSDP2 module the SDMA all-gather, if the patched torch SDMA ops are available and enabled
    (XDIT_FSDP_SDMA=1), or xfuser's pair all-gather (XDIT_FSDP_SDMA=pull, 2-rank FSDP groups)."""
    import os

    if os.environ.get("XDIT_FSDP_SDMA") == "pull":
        if not hasattr(module, "set_custom_all_gather"):
            return False
        module.set_custom_all_gather(PairAllGather())
        return True
    try:
        import torch.distributed._symmetric_memory._rocm_sdma as rs
    except ImportError:
        return False
    if not rs.enabled() or not hasattr(module, "set_custom_all_gather"):
        return False
    from xfuser.core.distributed import ipc_workspace

    ipc_workspace.install_torch_sdma_inbox()
    module.set_custom_all_gather(SdmaAllGather())
    return True
