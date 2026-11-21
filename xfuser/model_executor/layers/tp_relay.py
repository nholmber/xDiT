"""TP collectives relayed through the GPUs of the other TP group (XDIT_TP_RELAY=1), with the persistent FP8 GEMMs.

For layouts with two TP groups on one node running the same TP calls at the same time (U2 x TP4: groups
{0..3} and {4..7}), a TP group's own links carry only part of the node's bandwidth. Here every chunk a
rank sends to a TP peer goes in 1 + H pieces: one over the direct link and one through each GPU of the
other group (a helper), which receives it into a staging buffer and forwards it to the peer from a side
stream once it has landed. Each rank is the helper of the other group's transfers, symmetrically.

Transport: xDiT's SDMA transport (core/distributed/sdma/transport.py) over the world group. Copies are
registered once as plans and released per call; every plan writes one flag word per peer behind its
copies on that peer's engine (peers a plan does not target get a scratch word).

Protocol (no barriers): flag words are never reset; each op kind numbers its calls, every flag write of
call n writes n, and every wait is "word >= n". Buffers are doubled by call parity: a rank sends call
n + 2 into a parity only after it received every peer's call n + 1 data, which each peer (and helper)
sends only after it finished with call n's (stream order), so nobody writes a buffer still in use.
All ranks of the node make the same relayed calls in the same order (both TP groups run the same step).

Pull mode (XDIT_TP_RELAY_PULL=1, as ROCm/aiter#6236's relay: the helper only lends memory): the destination,
not the helper, moves the second hop. Per piece it waits on the helper's staging flags (remote, uncached flag
buffers) from a side stream, then its own engine for that helper pulls every sender's staged piece into place
and sets its local data words. Helpers issue nothing for the other group.

All-gather + GEMM (FP8): the row scales go first, direct; chunk c of every rank's rows lands in P = 1 + H
pieces, and the persistent kernel (layers/flydsl_tp_gemm.py, "ag") waits in-kernel for all P pieces of a
(chunk, rank) before its tiles. GEMM + reduce-scatter (FP8): the persistent kernel ("rs") writes partial
sums in send order and counts tiles per (chunk, dest); one side stream, per chunk, posts this rank's
pieces once counted and then forwards the other group's staged pieces of that chunk; each destination
reduces a chunk in fp32 once all pieces from all senders are in. Rows per rank must be whole 256-row tiles
(XDIT_TP_ROW_ALIGN=256); pieces are XDIT_TP_RELAY_CHUNKS - 1 equal runs of tiles and a last one with the
remainder (chunk_rows), so a tile count with no divisor (Wan: 37) still pipelines.
"""

import ctypes
import dataclasses
import functools
from typing import Dict, Optional

import torch
import torch.distributed as dist

import xfuser.envs as envs
from xfuser.logger import init_logger

logger = init_logger(__name__)

BLOCK_M = 256
_ALIGN = 256
_VALUES = 1 << 22  # call numbers available as flag values before the counters wrap
_relays: Dict[str, "TPRelay"] = {}
_side: Dict[int, torch.cuda.Stream] = {}  # helper forwards
_post: Dict[int, torch.cuda.Stream] = {}  # GEMM + reduce-scatter: this rank's chunks, once their tiles are counted
calls = {"all_gather_gemm": 0, "gemm_reduce_scatter": 0, "rs_then_ag": 0}


def enabled() -> bool:
    return envs.environment_variables["XDIT_TP_RELAY"]().strip() == "1"


@functools.lru_cache(maxsize=None)
def _wait_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def wait_words(F, IDX, n, target):
        """Spin (system-scope acquire) until F[IDX[i]] >= target for every i < n."""
        for i in range(n):
            w = tl.load(IDX + i)
            v = tl.atomic_add(F + w, 0, sem="acquire", scope="sys")
            while v < target:
                v = tl.atomic_add(F + w, 0, sem="acquire", scope="sys")

    return wait_words


@functools.lru_cache(maxsize=None)
def _reduce_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_chunk(RECV, OWN, OUT, sender_stride, me, n, W: tl.constexpr, BLOCK: tl.constexpr):
        """OUT[i] = bf16(sum over r < W in rank order of part_r[i]) in fp32: part_r = RECV + r * sender_stride
        (this chunk of sender r's partials), or OWN for r == me."""
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for r in tl.static_range(W):
            if r == me:
                v = tl.load(OWN + offs, mask=m, other=0.0)
            else:
                v = tl.load(RECV + r * sender_stride + offs, mask=m, other=0.0)
            acc += v.to(tl.float32)
        tl.store(OUT + offs, acc.to(tl.bfloat16), mask=m)

    return reduce_chunk


@functools.lru_cache(maxsize=None)
def _fp8_kernels():
    """Row x 256-column block FP8 (e4m3) quantization of this rank's partial sums for its peers, and the reduce
    that dequantizes the peers' parts (this rank's own part stays bf16)."""
    import triton
    import triton.language as tl

    @triton.jit
    def quant_chunk(P, Q, SC, rows, r0, nr, n_rows, me, N, NB, ROWS: tl.constexpr, BLOCK: tl.constexpr):
        """Rows r0 .. r0 + nr of every peer's block of the [W * rows, N] partials -> Q (fp8) and SC (fp32, one scale
        per row and BLOCK columns). ROWS rows per program, looping over the column blocks: few, fat programs, since
        this runs beside the persistent GEMM on the compute units it leaves free."""
        i = tl.program_id(0) * ROWS + tl.arange(0, ROWS)
        ok = i < n_rows
        di = i // nr
        dest = di + (di >= me).to(tl.int32)
        grow = (dest * rows + r0 + i % nr).to(tl.int64)
        cols = tl.arange(0, BLOCK)
        for b in range(0, NB):
            x = tl.load(P + grow[:, None] * N + b * BLOCK + cols[None, :], mask=ok[:, None], other=0.0).to(tl.float32)
            scale = tl.maximum(tl.max(tl.abs(x), axis=1) / 448.0, 1e-30)
            tl.store(
                Q + grow[:, None] * N + b * BLOCK + cols[None, :],
                (x / scale[:, None]).to(tl.float8e4nv),
                mask=ok[:, None],
            )
            tl.store(SC + grow * NB + b, scale, mask=ok)

    @triton.jit
    def reduce_chunk_fp8(
        RECV, RSC, OWN, OUT, sender_stride, sc_stride, me, n, N, NB, W: tl.constexpr, BLOCK: tl.constexpr
    ):
        """OUT[i] = bf16(sum over r < W in rank order): rank r's fp8 part times its block scale, or OWN for r == me."""
        offs = tl.program_id(0).to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        m = offs < n
        sc_off = (offs // N) * NB + (offs % N) // 256
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for r in tl.static_range(W):
            if r == me:
                v = tl.load(OWN + offs, mask=m, other=0.0).to(tl.float32)
            else:
                q = tl.load(RECV + r * sender_stride + offs, mask=m, other=0.0).to(tl.float32)
                v = q * tl.load(RSC + r * sc_stride + sc_off, mask=m, other=0.0)
            acc += v
        tl.store(OUT + offs, acc.to(tl.bfloat16), mask=m)

    return quant_chunk, reduce_chunk_fp8


@dataclasses.dataclass
class RowEpilogue:
    """The row-wise work between a reduce-scatter and the next all-gather in a Wan TP block, for one fused kernel
    (reduce -> + bias -> residual update -> LayerNorm -> modulation -> FP8 row quantization): y = reduced + bias;
    h = residual + y (fp32 times gate, or in bf16); out = LayerNorm(h) [* (1 + scale) + shift] (fp32). Tensors are
    [N] (fp32 except the bf16 bias) or None; `residual` is this rank's [rows, N] bf16 and `new_residual` receives h."""

    residual: torch.Tensor
    new_residual: torch.Tensor
    bias: Optional[torch.Tensor]
    gate: Optional[torch.Tensor]
    norm_w: Optional[torch.Tensor]
    norm_b: Optional[torch.Tensor]
    eps: float
    mod_scale: Optional[torch.Tensor]
    mod_shift: Optional[torch.Tensor]


@functools.lru_cache(maxsize=None)
def _reduce_norm_quant_kernel():
    import triton
    import triton.language as tl

    @triton.jit
    def reduce_norm_quant(
        RECV,
        RSC,
        OWN,
        BIAS,
        RES,
        GATE,
        NW,
        NB_,
        MS,
        MT,
        H,
        Q,
        S,
        sender_stride,
        sc_stride,
        me,
        r0,
        N,
        NBLK,
        eps,
        W: tl.constexpr,
        FP8: tl.constexpr,
        HAS_BIAS: tl.constexpr,
        HAS_GATE: tl.constexpr,
        HAS_NW: tl.constexpr,
        HAS_MOD: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One row of a reduced piece: the senders' parts (fp8 x block scale, or bf16; own part bf16) summed in fp32
        in rank order and rounded to bf16 as the reduce does, then the residual update, LayerNorm, modulation and
        row-wise FP8 quantization with the roundings of the unfused block. Writes H (new residual), Q, S."""
        row = tl.program_id(0)
        cols = tl.arange(0, BLOCK)
        m = cols < N
        off = row.to(tl.int64) * N + cols
        acc = tl.zeros((BLOCK,), dtype=tl.float32)
        for r in tl.static_range(W):
            if r == me:
                v = tl.load(OWN + off, mask=m, other=0.0).to(tl.float32)
            elif FP8:
                q = tl.load(RECV + r * sender_stride + off, mask=m, other=0.0).to(tl.float32)
                v = q * tl.load(RSC + r * sc_stride + row.to(tl.int64) * NBLK + cols // 256, mask=m, other=0.0)
            else:
                v = tl.load(RECV + r * sender_stride + off, mask=m, other=0.0).to(tl.float32)
            acc += v
        y = acc.to(tl.bfloat16)
        if HAS_BIAS:
            y = (y.to(tl.float32) + tl.load(BIAS + cols, mask=m, other=0.0).to(tl.float32)).to(tl.bfloat16)
        grow = (r0 + row).to(tl.int64) * N + cols
        res = tl.load(RES + grow, mask=m, other=0.0)
        if HAS_GATE:
            h = (res.to(tl.float32) + y.to(tl.float32) * tl.load(GATE + cols, mask=m, other=0.0)).to(tl.bfloat16)
        else:
            h = (res.to(tl.float32) + y.to(tl.float32)).to(tl.bfloat16)
        tl.store(H + grow, h, mask=m)
        hf = tl.where(m, h.to(tl.float32), 0.0)
        mean = tl.sum(hf, axis=0) / N
        d = tl.where(m, hf - mean, 0.0)
        n = d * tl.rsqrt(tl.sum(d * d, axis=0) / N + eps)
        if HAS_NW:
            n = n * tl.load(NW + cols, mask=m, other=0.0) + tl.load(NB_ + cols, mask=m, other=0.0)
        if HAS_MOD:
            n = n * (1 + tl.load(MS + cols, mask=m, other=0.0)) + tl.load(MT + cols, mask=m, other=0.0)
        o = tl.where(m, n.to(tl.bfloat16).to(tl.float32), 0.0)
        scale = tl.maximum(tl.max(tl.abs(o), axis=0), 1e-12) / 448.0
        tl.store(Q + grow, (o / scale).to(tl.float8e4nv), mask=m)
        tl.store(S + r0 + row, scale)

    return reduce_norm_quant


def chunk_rows(rows: int, C: int):
    """[(first row, rows)] of C chunks of whole BLOCK_M-row tiles: C - 1 equal ones, the last also takes the remainder
    (as the persistent kernel splits a rank's row blocks)."""
    base = rows // BLOCK_M // C * BLOCK_M
    return [(c * base, base if c < C - 1 else rows - (C - 1) * base) for c in range(C)]


def _pieces(nbytes: int, helpers: int, direct_frac: float):
    """[(offset, size)] of a chunk's pieces: the direct one first, then one per helper; 256-byte aligned."""
    d = int(nbytes * direct_frac) // _ALIGN * _ALIGN
    hp = ((nbytes - d) // helpers) // _ALIGN * _ALIGN
    sizes = [d] + [hp] * (helpers - 1) + [nbytes - d - hp * (helpers - 1)]
    return [(sum(sizes[:j]), s) for j, s in enumerate(sizes)]


class TPRelay:
    """The relay state of one TP group: the world transport, the helpers, buffers and plans by shape."""

    def __init__(self, group_name: str):
        from xfuser.core.distributed.parallel_state import get_world_group
        from xfuser.core.distributed.sdma import transport

        self.T = transport
        group = dist.distributed_c10d._resolve_process_group(group_name)
        self.ranks = dist.get_process_group_ranks(group)
        self.Wt = len(self.ranks)
        self.w = transport.group_for(get_world_group().device_group)
        world = self.w.world
        if world != 2 * self.Wt:
            raise ValueError(f"the TP relay needs two TP groups of {self.Wt} on {world} GPUs")
        everyone = [None] * world
        dist.all_gather_object(everyone, tuple(self.ranks), group=self.w.process_group)
        groups = sorted(set(everyone))
        if len(groups) != 2 or any(len(g) != self.Wt for g in groups):
            raise ValueError(f"the TP relay needs two disjoint TP groups of {self.Wt}; got {groups}")
        self.me = self.ranks.index(self.w.rank)
        # helper j of this group is rank j of the other one; this rank is helper `me` of the other group
        self.helpers = list(next(g for g in groups if g != tuple(self.ranks)))
        self.H = len(self.helpers)
        self.P = 1 + self.H
        self.direct_frac = float(envs.environment_variables["XDIT_TP_RELAY_DIRECT"]())
        rs_direct = envs.environment_variables["XDIT_TP_RELAY_RS_DIRECT"]().strip()
        self.direct_frac_rs = float(rs_direct) if rs_direct else self.direct_frac
        self.want_chunks = int(envs.environment_variables["XDIT_TP_RELAY_CHUNKS"]())
        self.rs_fp8 = envs.environment_variables["XDIT_TP_RELAY_RS_FP8"]().strip() == "1"
        self.n = {"ag": 0, "rs": 0}  # calls per op kind (flag values)
        import os

        self.pull = os.environ.get("XDIT_TP_RELAY_PULL", "0") == "1"
        self._plans: Dict[tuple, dict] = {}
        dev = torch.device("cuda", torch.cuda.current_device())
        self.values = torch.arange(_VALUES, dtype=torch.int32, device=dev)  # flag source of call n: values[n]
        logger.info(f"TP relay ({'pull' if self.pull else 'forward'}) for group {self.ranks} through {self.helpers}")

    def chunks(self, rows: int) -> Optional[int]:
        """Pieces for `rows` rows per rank: XDIT_TP_RELAY_CHUNKS, at most one per 256-row tile and at most what fits
        the transport's sends per call (all-gather: 1 + C * (1 + Wt)); None unless rows are whole tiles."""
        if rows % BLOCK_M:
            return None
        cap = (self.T._TRIGGERS - 1) // (1 + self.Wt)
        return max(1, min(self.want_chunks, rows // BLOCK_M, cap))

    def _next(self, kind: str) -> int:
        self.n[kind] += 1
        if self.n[kind] >= _VALUES:
            raise RuntimeError("TP relay call counter wrapped")
        return self.n[kind]

    def base(self, buf, rank: int) -> int:
        return buf.ptr if rank == self.w.rank else buf.peer_bases[self.w.peers.index(rank)]

    def _flags_layout(self, C: int):
        """Word offsets in a flag buffer: data [C][Wt][P], scales [Wt], stage [C][Wt], scratch."""
        data = 0
        scale = data + C * self.Wt * self.P
        stage = scale + self.Wt
        scratch = stage + C * self.Wt
        return data, scale, stage, scratch, scratch + 1

    def _plan(self, copies, flags):
        return self.T.Plan(self.w, copies, flags)

    @staticmethod
    def _free(entry):
        from xfuser.core.distributed.sdma import transport

        for pp in entry["par"]:
            plans = [*pp["chunks"], *(x for row in pp["fwd"] for x in row)]
            if "scale" in pp:
                plans.append(pp["scale"])
            for plan in plans:
                plan.free()
            for pid in pp.get("pull", []):
                transport.load().sx_plan_free(pid)

    def _flags_buffer(self, name: str, nbytes: int):
        # Pull mode polls the helpers' flag words from the destination: uncached, so remote polls see the writes.
        return self.w._buffer(name, nbytes, uncached=True) if self.pull else self.w.buffer(name, nbytes)

    def _pull_plans(self, C: int, pieces, stage_at, dst_at, data: int, f) -> list:
        """Pull mode, per chunk c: one raw plan whose entries pull every sender's staged piece of chunk c from each
        helper (on this rank's engine for that helper) into this rank's receive buffer, each entry then setting
        this rank's data word of (c, sender, piece)."""
        from xfuser.core.distributed.sdma.pull_relay import _plan

        engines_of = self.w.engine_plan()
        me, Wt = self.me, self.Wt
        out = []
        for c in range(C):
            engines, copies, flags = [], [], []
            for j, q in enumerate(self.helpers):
                off, size = pieces[c][1 + j]
                for i in range(Wt):
                    if i == me:
                        continue
                    engines.append(engines_of[q])
                    copies.append([(stage_at(q, i, me, c), dst_at(i, c, off), size)])
                    flags.append(self.base(f, self.w.rank) + (data + (c * Wt + i) * self.P + 1 + j) * 4)
            out.append(_plan(engines, copies, flags))
        return out

    def _pull_waits(self, C: int, f, stage: int, nwords: int):
        """Pull mode: each helper's flag words as a tensor here, and per chunk the stage words to wait on there."""
        from xfuser.core.distributed.sdma import transport

        dev = self.values.device
        remote = [transport.device_bytes(self.base(f, q), nwords * 4, dev).view(torch.int32) for q in self.helpers]
        idx = [
            torch.tensor([stage + c * self.Wt + i for i in range(self.Wt) if i != self.me], dtype=torch.int32, device=dev)
            for c in range(C)
        ]
        return remote, idx

    def _pull_chunk(self, call, e, pp, c: int, n: int, flag_src: int, side) -> None:
        """Pull mode: on `side`, wait until every sender's piece c is staged at every helper, then pull them."""
        from xfuser.core.distributed.sdma import transport

        lib = transport.load()
        wait = _wait_kernel()
        remote, idx = e["pull_waits"]
        with torch.cuda.stream(side):
            for flags in remote:
                wait[(1,)](flags, idx[c], idx[c].numel(), n)
        k = call._sends
        if k >= transport._TRIGGERS:
            raise ValueError(f"at most {transport._TRIGGERS} sends per call")
        transport._check(lib.sx_plan_issue(call.slot, k, pp["pull"][c], 0, flag_src), "sx_plan_issue")
        transport._check(lib.sx_release(call.slot, k, ctypes.c_void_p(side.cuda_stream)), "sx_release")
        call._sends += 1

    # -- all-gather + GEMM ------------------------------------------------------------------------------
    def _ag(self, rows: int, K: int, C: int, chunk_scales: bool = False) -> dict:
        """Plans of a relayed all-gather. chunk_scales: each piece's row scales travel with its direct copy (send
        source = [rows, K] codes then [rows] fp32 scales), so the GEMM's per-piece data wait covers them and no
        scale plan is sent (a piece-by-piece producer has no scales until its last piece)."""
        key = ("ag", rows, K, C, chunk_scales)
        w, Wt, H, P = self.w, self.Wt, self.H, self.P
        bounds = chunk_rows(rows, C)
        coff = [r0 * K for r0, _ in bounds]  # byte offset of chunk c in a rank's rows (fp8)
        pieces = [_pieces(n * K, H, self.direct_frac) for _, n in bounds]
        slot = max(s for pc in pieces for _, s in pc[1:])
        xh, sh, sth = Wt * rows * K, Wt * rows * 4, Wt * Wt * C * slot  # one parity's bytes
        bufs = {
            "x": w.buffer("tp_relay_ag_x", 2 * xh),
            "s": w.buffer("tp_relay_ag_s", 2 * sh),
            "stage": w.buffer("tp_relay_ag_stage", 2 * sth),  # [parity][client sender][client dest][c]
            "flags": self._flags_buffer("tp_relay_ag_flags", self._flags_layout(C)[-1] * 4),
        }
        cached = self._plans.get(key)
        if cached is not None and all(cached["bufs"][k] is b and b.live for k, b in bufs.items()):
            return cached
        data, scale, stage, scratch, _ = self._flags_layout(C)
        me, ranks, helpers, peers = self.me, self.ranks, self.helpers, w.peers
        f = bufs["flags"]

        def flag(rank, word):
            return self.base(f, rank) + word * 4

        par_plans = []
        for par in range(2):

            def x_at(rank, slot_rank, c, off, par=par):
                return self.base(bufs["x"], rank) + par * xh + slot_rank * rows * K + coff[c] + off

            def stage_at(helper_rank, sender, dest, c, par=par):
                return self.base(bufs["stage"], helper_rank) + par * sth + ((sender * Wt + dest) * C + c) * slot

            copies, flags = [], []  # row scales: to every TP peer, direct
            for q in peers:
                if q in ranks:
                    copies.append([(0, self.base(bufs["s"], q) + par * sh + me * rows * 4, rows * 4)])
                    flags.append(flag(q, scale + me))
                else:
                    copies.append([])
                    flags.append(flag(q, scratch))
            scale_plan = self._plan(copies, flags)
            chunk_plans = []  # chunk c: the direct piece to each TP peer, every peer's helper piece to each helper
            for c in range(C):
                copies, flags = [], []
                for q in peers:
                    if q in ranks:
                        off, size = pieces[c][0]
                        direct = [(coff[c] + off, x_at(q, me, c, off), size)]
                        if chunk_scales:  # this piece's row scales, on the same engine before the flag
                            r0, nr = bounds[c]
                            s_dst = self.base(bufs["s"], q) + par * sh + (me * rows + r0) * 4
                            direct.append((rows * K + r0 * 4, s_dst, nr * 4))
                        copies.append(direct)
                        flags.append(flag(q, data + (c * Wt + me) * P))
                    else:
                        off, size = pieces[c][1 + helpers.index(q)]
                        copies.append([(coff[c] + off, stage_at(q, me, d, c), size) for d in range(Wt) if d != me])
                        flags.append(flag(q, stage + c * Wt + me))
                chunk_plans.append(self._plan(copies, flags))
            if self.pull:
                pull = self._pull_plans(
                    C, pieces, stage_at, lambda i, c, off, x_at=x_at: x_at(w.rank, i, c, off), data, f
                )
                par_plans.append({"scale": scale_plan, "chunks": chunk_plans, "fwd": [], "pull": pull})
                continue
            # as helper `me` of the other group: forward client sender i's pieces of chunk c to each client dest
            fwd = []
            for c in range(C):
                off, size = pieces[c][1 + me]
                row = []
                for i in range(Wt):
                    copies, flags = [], []
                    for q in peers:
                        if q in helpers and helpers.index(q) != i:
                            d = helpers.index(q)
                            src = stage_at(w.rank, i, d, c) - bufs["stage"].ptr
                            copies.append([(src, x_at(q, i, c, off), size)])
                            flags.append(flag(q, data + (c * Wt + i) * P + 1 + me))
                        else:
                            copies.append([])
                            flags.append(flag(q, scratch))
                    row.append(self._plan(copies, flags))
                fwd.append(row)
            par_plans.append({"scale": scale_plan, "chunks": chunk_plans, "fwd": fwd})
        if cached is not None:
            self._free(cached)
        dev = self.values.device
        entry = {
            "par": par_plans,
            "bufs": bufs,
            "halves": (xh, sh, sth),
            # per-shape index lists (a per-call host->device copy would stall the host)
            "scale_idx": torch.tensor([scale + i for i in range(Wt) if i != me], dtype=torch.int32, device=dev),
            "stage_idx": [
                torch.tensor([stage + c * Wt + i], dtype=torch.int32, device=dev) for c in range(C) for i in range(Wt)
            ],
            "chunk_scales": chunk_scales,
            "pull_waits": self._pull_waits(C, f, stage, self._flags_layout(C)[-1]) if self.pull else None,
            "own_words": torch.tensor(
                [data + (c * Wt + me) * P + k for c in range(C) for k in range(P)] + [scale + me],
                dtype=torch.long,
                device=dev,
            ),
        }
        self._plans[key] = entry
        return entry

    def _ag_begin(self, rows: int, K: int, C: int, call, cur, chunk_scales: bool = False) -> dict:
        """Open a relayed all-gather of `rows` x K fp8 rows in C pieces on `call`: buffers, this rank's flag words,
        the helper-side forwarders (side stream). The pieces go out with _ag_send_chunk / _ag_send_scales."""
        Wt = self.Wt
        e = self._ag(rows, K, C, chunk_scales)
        n = self._next("ag")
        par = n % 2
        xh, sh, _ = e["halves"]
        nwords = self._flags_layout(C)[-1]
        dev = cur.device
        side = _side.setdefault(dev.index, torch.cuda.Stream(device=dev, priority=-1))
        flags = e["bufs"]["flags"].local.view(torch.int32)[:nwords]
        st = {
            "e": e,
            "n": n,
            "pp": e["par"][par],
            "rows": rows,
            "K": K,
            "C": C,
            "flags": flags,
            "side": side,
            "x_in": e["bufs"]["x"].local[par * xh : (par + 1) * xh].view(Wt, rows * K),
            "s_in": e["bufs"]["s"].local[par * sh : (par + 1) * sh].view(torch.float32).view(Wt, rows),
            "flag_src": self.values[n : n + 1].data_ptr(),
            "call": call,
        }
        # own flag words for this call (no reset: peers' words only ever grow)
        flags.index_fill_(0, e["own_words"], n)
        side.wait_stream(cur)
        return st

    def _ag_forward_chunk(self, st, c: int) -> None:
        """Helper side of piece c: forward each client sender's piece once it has landed in this rank's staging
        (side stream). Engines run copies in host issue order, so issue a piece's forwards after this rank's own
        send of that piece: forwards of a later piece ahead of it would wait for peers' sends queued behind their
        own forwards of that piece (a cycle)."""
        e, n, Wt, side = st["e"], st["n"], self.Wt, st["side"]
        if self.pull:
            self._pull_chunk(st["call"], e, st["pp"], c, n, st["flag_src"], side)
            return
        stage_all = e["bufs"]["stage"].local  # forward plans hold offsets from the buffer start (parity included)
        wait = _wait_kernel()
        with torch.cuda.stream(side):
            for i in range(Wt):
                wait[(1,)](st["flags"], e["stage_idx"][c * Wt + i], 1, n)
                st["call"].send(st["pp"]["fwd"][c][i], stage_all, side, st["flag_src"])

    def _ag_send_chunk(self, st, c: int, xq_rows, cur) -> None:
        """Piece c of this rank's rows: `xq_rows` is the full [rows, K] fp8 tensor whose piece c is ready on `cur`."""
        r0, nr = chunk_rows(st["rows"], st["C"])[c]
        K = st["K"]
        st["x_in"][self.me][r0 * K : (r0 + nr) * K].copy_(xq_rows[r0 : r0 + nr].view(torch.uint8).view(-1))
        src = xq_rows
        if st["e"]["chunk_scales"]:  # xq_rows is the codes view of a [codes | scales] send buffer
            src, rows = st["send_buf"], st["rows"]
            st["s_in"][self.me][r0 : r0 + nr].copy_(src[rows * K :].view(torch.float32)[r0 : r0 + nr])
        st["call"].send(st["pp"]["chunks"][c], src, cur, st["flag_src"])

    def _ag_send_scales(self, st, xs_rows, cur) -> None:
        """This rank's row scales ([rows] fp32, ready on `cur`), to every TP peer."""
        st["s_in"][self.me].copy_(xs_rows)
        st["call"].send(st["pp"]["scale"], xs_rows, cur, st["flag_src"])

    def _ag_gemm(self, st, w_t, w_s, cur) -> torch.Tensor:
        """The persistent GEMM over the gathered rows (peers' pieces waited for in-kernel): [Wt * rows, N] bf16."""
        import flydsl.expr as fx
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        from xfuser.model_executor.layers.flydsl_tp_gemm import compile_persistent_ag_gemm_fp8

        e, rows, K, C, Wt = st["e"], st["rows"], st["K"], st["C"], self.Wt
        N = w_t.shape[1]
        data = self._flags_layout(C)[0]
        out = torch.empty(Wt * rows, N, dtype=torch.bfloat16, device=cur.device)
        if not e["chunk_scales"]:  # else each piece's scales came with its data (waited for in-kernel)
            _wait_kernel()[(1,)](st["flags"], e["scale_idx"], e["scale_idx"].numel(), st["n"])
        exe = compile_persistent_ag_gemm_fp8(K=K, BLOCK_M=BLOCK_M, BLOCK_N=256)
        # leave CUs for the forwarders on the side stream: other ranks wait for what they forward
        progs = torch.cuda.get_device_properties(cur.device).multi_processor_count * 7 // 8
        _run_compiled(
            exe,
            st["x_in"][0][: rows * K].view(torch.int8),  # rank-0-sized views: the kernel rebases per rank
            w_t.t().view(torch.int8).reshape(-1),
            out.view(-1)[: rows * N],
            st["s_in"][0][:rows],
            w_s.reshape(-1).float().contiguous(),
            e["bufs"]["flags"].ptr,
            st["n"],
            self.me,
            Wt,
            rows,
            chunk_rows(rows, C)[0][1],
            C,
            rows,
            rows,
            data,
            self.P,
            N,
            progs,
            fx.Stream(cur),
        )
        return out

    def all_gather_gemm_fp8(self, xq, xs, w_t, w_s, chunks: int, keep: bool = False):
        """xq [rows, K] fp8, xs [rows, 1] fp32, w_t [K, N] column-major, w_s [1, N] -> [Wt * rows, N] bf16.

        keep: also return the gathered rows [Wt * rows, K] and scales [Wt * rows, 1], views of this call's
        receive buffers. They hold until this rank's next-but-one relayed all-gather is issued (peers write
        that parity again only after this rank's next all-gather has sent, i.e. after work queued before it)."""
        rows, K = xq.shape
        cur = torch.cuda.current_stream()
        xq = xq.contiguous()
        xs = xs.reshape(rows).float().contiguous()
        call = self.w.begin()
        st = self._ag_begin(rows, K, chunks, call, cur)
        self._ag_send_scales(st, xs, cur)
        for c in range(chunks):
            self._ag_send_chunk(st, c, xq, cur)
        for c in range(chunks):
            self._ag_forward_chunk(st, c)
        out = self._ag_gemm(st, w_t, w_s, cur)
        cur.wait_stream(st["side"])
        call.end()
        xq.record_stream(st["side"])
        calls["all_gather_gemm"] += 1
        if keep:
            Wt = self.Wt
            return out, st["x_in"].view(xq.dtype).view(Wt * rows, K), st["s_in"].view(Wt * rows, 1)
        return out

    # -- GEMM + reduce-scatter ------------------------------------------------------------------------------
    def _rs(self, rows: int, N: int, C: int, fp8: bool) -> dict:
        """Plans of GEMM + reduce-scatter: partial sums as bf16, or (fp8) as e4m3 codes with one fp32 scale per row
        and 256 columns (the scales go with the direct piece). The send buffer holds [W * rows, N] codes, then
        [W * rows, N / 256] scales."""
        key = ("rs", rows, N, C, fp8)
        w, Wt, H, P = self.w, self.Wt, self.H, self.P
        esz, NB = (1, N // 256) if fp8 else (2, 0)
        bounds = chunk_rows(rows, C)
        pieces = [_pieces(n * N * esz, H, self.direct_frac_rs) for _, n in bounds]  # chunk c of partial sums
        slot = max(s for pc in pieces for _, s in pc[1:])
        rh, sth, rsh = Wt * rows * N * esz, Wt * Wt * C * slot, Wt * rows * NB * 4
        bufs = {
            "recv": w.buffer("tp_relay_rs_recv", 2 * rh),  # [parity][sender][rows] at each destination
            "stage": w.buffer("tp_relay_rs_stage", 2 * sth),
            "flags": self._flags_buffer("tp_relay_rs_flags", self._flags_layout(C)[-1] * 4),
        }
        if fp8:
            bufs["rscale"] = w.buffer("tp_relay_rs_rscale", 2 * rsh)  # [parity][sender][rows][N / 256]
        cached = self._plans.get(key)
        if cached is not None and all(cached["bufs"][k] is b and b.live for k, b in bufs.items()):
            return cached
        data, _, stage, scratch, _ = self._flags_layout(C)
        me, ranks, helpers, peers = self.me, self.ranks, self.helpers, w.peers
        f = bufs["flags"]

        def flag(rank, word):
            return self.base(f, rank) + word * 4

        def src(dest, c):  # this rank's partials for dest's chunk c: rows dest * rows + its first row, of [S, N]
            return (dest * rows + bounds[c][0]) * N * esz

        def src_scale(dest, c):  # its scales, after the S x N codes in the send buffer
            return Wt * rows * N + (dest * rows + bounds[c][0]) * NB * 4

        par_plans = []
        for par in range(2):

            def recv_at(rank, sender, c, off, par=par):
                return self.base(bufs["recv"], rank) + par * rh + (sender * rows + bounds[c][0]) * N * esz + off

            def rscale_at(rank, sender, c, par=par):
                return self.base(bufs["rscale"], rank) + par * rsh + (sender * rows + bounds[c][0]) * NB * 4

            def stage_at(helper_rank, sender, dest, c, par=par):
                return self.base(bufs["stage"], helper_rank) + par * sth + ((sender * Wt + dest) * C + c) * slot

            chunk_plans = []
            for c in range(C):
                copies, flags = [], []
                for q in peers:
                    if q in ranks:
                        d = ranks.index(q)
                        off, size = pieces[c][0]
                        direct = [(src(d, c) + off, recv_at(q, me, c, off), size)]
                        if fp8:  # the chunk's scales, on the same engine before the flag
                            direct.append((src_scale(d, c), rscale_at(q, me, c), bounds[c][1] * NB * 4))
                        copies.append(direct)
                        flags.append(flag(q, data + (c * Wt + me) * P))
                    else:
                        off, size = pieces[c][1 + helpers.index(q)]
                        copies.append([(src(d, c) + off, stage_at(q, me, d, c), size) for d in range(Wt) if d != me])
                        flags.append(flag(q, stage + c * Wt + me))
                chunk_plans.append(self._plan(copies, flags))
            if self.pull:
                pull = self._pull_plans(
                    C, pieces, stage_at, lambda i, c, off, recv_at=recv_at: recv_at(w.rank, i, c, off), data, f
                )
                par_plans.append({"chunks": chunk_plans, "fwd": [], "pull": pull})
                continue
            fwd = []
            for c in range(C):
                off, size = pieces[c][1 + me]
                row = []
                for i in range(Wt):
                    copies, flags = [], []
                    for q in peers:
                        if q in helpers and helpers.index(q) != i:
                            d = helpers.index(q)
                            s_off = stage_at(w.rank, i, d, c) - bufs["stage"].ptr
                            copies.append([(s_off, recv_at(q, i, c, off), size)])
                            flags.append(flag(q, data + (c * Wt + i) * P + 1 + me))
                        else:
                            copies.append([])
                            flags.append(flag(q, scratch))
                    row.append(self._plan(copies, flags))
                fwd.append(row)
            par_plans.append({"chunks": chunk_plans, "fwd": fwd})
        if cached is not None:
            self._free(cached)
        dev = self.values.device
        peers_of = [i for i in range(Wt) if i != me]
        entry = {
            "par": par_plans,
            "bufs": bufs,
            "halves": (rh, sth, rsh),
            "fp8": fp8,
            "NB": NB,
            "tiles": [n // BLOCK_M * ((N + 255) // 256) for _, n in bounds],  # GEMM tiles per (chunk c, dest)
            "bounds": bounds,
            "cnt_idx": [torch.tensor([c * Wt + p for p in peers_of], dtype=torch.int32, device=dev) for c in range(C)],
            "stage_idx": [
                torch.tensor([stage + c * Wt + i], dtype=torch.int32, device=dev) for c in range(C) for i in range(Wt)
            ],
            "data_idx": [
                torch.tensor(
                    [data + (c * Wt + s) * P + k for s in peers_of for k in range(P)], dtype=torch.int32, device=dev
                )
                for c in range(C)
            ],
            "cnt": torch.zeros(C * Wt, dtype=torch.int32, device=dev),  # tile counters [c][dest], zeroed per call
            "pull_waits": self._pull_waits(C, f, stage, self._flags_layout(C)[-1]) if self.pull else None,
        }
        self._plans[key] = entry
        return entry

    def _rs_begin(self, xq, xs, w_t, w_s, C: int, call, cur) -> dict:
        """Start a relayed GEMM + reduce-scatter on `call`: the poster / forwarder streams' sends and the persistent
        GEMM. Its pieces are reduced with _rs_reduce_chunk and closed with _rs_end."""
        import flydsl.expr as fx
        from aiter.ops.flydsl.kernels.tensor_shim import _run_compiled

        from xfuser.model_executor.layers.flydsl_tp_gemm import compile_persistent_ag_gemm_fp8

        S, K = xq.shape
        N = w_t.shape[1]
        Wt = self.Wt
        rows = S // Wt
        fp8 = self.rs_fp8 and N % 256 == 0
        e = self._rs(rows, N, C, fp8)
        bounds, NB = e["bounds"], e["NB"]
        n = self._next("rs")
        par = n % 2
        pp = e["par"][par]
        rh, sth, rsh = e["halves"]
        nwords = self._flags_layout(C)[-1]
        dev = xq.device
        side = _side.setdefault(dev.index, torch.cuda.Stream(device=dev, priority=-1))
        flags = e["bufs"]["flags"].local.view(torch.int32)[:nwords]
        recv_flat = e["bufs"]["recv"].local[par * rh : (par + 1) * rh]
        recv_flat = recv_flat.view(torch.float8_e4m3fn if fp8 else torch.bfloat16).view(Wt, rows * N)
        stage_all = e["bufs"]["stage"].local  # forward plans hold offsets from the buffer start (parity included)
        flag_src = self.values[n : n + 1].data_ptr()
        cnt = e["cnt"]
        partials = torch.empty(S, N, dtype=torch.bfloat16, device=dev)
        st = {
            "e": e,
            "n": n,
            "fp8": fp8,
            "rows": rows,
            "N": N,
            "NB": NB,
            "bounds": bounds,
            "flags": flags,
            "recv_flat": recv_flat,
            "partials": partials,
            "send": partials,
            "side": side,
            "out": torch.empty(rows, N, dtype=torch.bfloat16, device=dev),
        }
        if fp8:  # the peers' partial sums go out as e4m3 codes + scales, quantized per chunk on the poster stream
            quant, _ = _fp8_kernels()
            send = st["send"] = torch.empty(S * N + S * NB * 4, dtype=torch.uint8, device=dev)
            q8 = send[: S * N].view(torch.float8_e4m3fn)
            sc = send[S * N :].view(torch.float32)
            st["rscale"] = (
                e["bufs"]["rscale"].local[par * rsh : (par + 1) * rsh].view(torch.float32).view(Wt, rows * NB)
            )
        cnt.zero_()
        post = st["post"] = _post.setdefault(dev.index, torch.cuda.Stream(device=dev, priority=-1))
        post.wait_stream(cur)  # counters zeroed before the poster reads them
        side.wait_stream(cur)
        wait = _wait_kernel()
        # Each engine runs its copies in issue (host) order, so issue in expected-ready order: this rank's chunk
        # c (poster stream, once its tiles are counted), then the other group's staged pieces of chunk c (forwarder
        # stream). Separate streams: a forwarder's wait never holds back the next post's release, nor vice versa.
        for c in range(C):
            with torch.cuda.stream(post):
                wait[(1,)](cnt, e["cnt_idx"][c], e["cnt_idx"][c].numel(), e["tiles"][c])
                if fp8:
                    r0, nr = bounds[c]
                    n_rows = (Wt - 1) * nr
                    quant[((n_rows + 7) // 8,)](
                        partials, q8, sc, rows, r0, nr, n_rows, self.me, N, NB, ROWS=8, BLOCK=256, num_warps=8
                    )
                call.send(pp["chunks"][c], st["send"], post, flag_src)
            if self.pull:
                self._pull_chunk(call, e, pp, c, n, flag_src, side)
                continue
            with torch.cuda.stream(side):
                for i in range(Wt):
                    wait[(1,)](flags, e["stage_idx"][c * Wt + i], 1, n)
                    call.send(pp["fwd"][c][i], stage_all, side, flag_src)
        exe = compile_persistent_ag_gemm_fp8(K=K, BLOCK_M=BLOCK_M, BLOCK_N=256, mode="rs")
        # CUs left free: the poster's / forwarders' spin kernels and, with FP8 partials, the quantization
        free = int(envs.environment_variables["XDIT_TP_RELAY_RS_FREE_CU_EIGHTHS"]()) if fp8 else 1
        progs = torch.cuda.get_device_properties(dev).multi_processor_count * (8 - free) // 8
        _run_compiled(
            exe,
            xq.contiguous().view(torch.int8).view(-1)[: rows * K],
            w_t.t().view(torch.int8).reshape(-1),
            partials.view(-1)[: rows * N],
            xs.reshape(-1).float().contiguous()[:rows],
            w_s.reshape(-1).float().contiguous(),
            cnt.data_ptr(),
            1,
            self.me,
            Wt,
            rows,
            bounds[0][1],
            C,
            rows,
            rows,
            0,
            1,
            N,
            progs,
            fx.Stream(cur),
        )
        return st

    def _rs_reduce_chunk(self, st, c: int) -> torch.Tensor:
        """Wait for every sender's piece c and reduce it (fp32, rank order) on the current stream: rows of `out`."""
        e, N, NB, rows, Wt = st["e"], st["N"], st["NB"], st["rows"], self.Wt
        r0, nr = st["bounds"][c]
        nel = nr * N
        out = st["out"]
        own = st["partials"].view(Wt, rows * N)[self.me]
        RB = 4096
        _wait_kernel()[(1,)](st["flags"], e["data_idx"][c], e["data_idx"][c].numel(), st["n"])
        if st["fp8"]:
            _, reduce_fp8 = _fp8_kernels()
            reduce_fp8[((nel + RB - 1) // RB,)](
                st["recv_flat"][0, r0 * N :],
                st["rscale"][0, r0 * NB :],
                own[r0 * N :],
                out[r0:],
                rows * N,
                rows * NB,
                self.me,
                nel,
                N,
                NB,
                W=Wt,
                BLOCK=RB,
            )
        else:
            _reduce_kernel()[((nel + RB - 1) // RB,)](
                st["recv_flat"][0, r0 * N :], own[r0 * N :], out[r0:], rows * N, self.me, nel, W=Wt, BLOCK=RB
            )
        return out[r0 : r0 + nr]

    def _rs_reduce_norm_quant_chunk(self, st, c: int, ep: RowEpilogue, nq, ns) -> None:
        """_rs_reduce_chunk fused with a RowEpilogue and the next all-gather's FP8 row quantization (one kernel)."""
        import triton

        e, N, NB, rows, Wt = st["e"], st["N"], st["NB"], st["rows"], self.Wt
        r0, nr = st["bounds"][c]
        own = st["partials"].view(Wt, rows * N)[self.me]
        _wait_kernel()[(1,)](st["flags"], e["data_idx"][c], e["data_idx"][c].numel(), st["n"])
        dummy = ns  # any pointer, for the inputs a constexpr switches off
        _reduce_norm_quant_kernel()[(nr,)](
            st["recv_flat"][0, r0 * N :],
            st["rscale"][0, r0 * NB :] if st["fp8"] else dummy,
            own[r0 * N :],
            ep.bias if ep.bias is not None else dummy,
            ep.residual,
            ep.gate if ep.gate is not None else dummy,
            ep.norm_w if ep.norm_w is not None else dummy,
            ep.norm_b if ep.norm_b is not None else dummy,
            ep.mod_scale if ep.mod_scale is not None else dummy,
            ep.mod_shift if ep.mod_shift is not None else dummy,
            ep.new_residual,
            nq,
            ns,
            rows * N,
            rows * NB,
            self.me,
            r0,
            N,
            NB,
            ep.eps,
            W=Wt,
            FP8=st["fp8"],
            HAS_BIAS=ep.bias is not None,
            HAS_GATE=ep.gate is not None,
            HAS_NW=ep.norm_w is not None,
            HAS_MOD=ep.mod_scale is not None,
            BLOCK=triton.next_power_of_2(N),
            num_warps=16,
        )

    def _rs_end(self, st, cur) -> None:
        cur.wait_stream(st["post"])
        cur.wait_stream(st["side"])
        st["partials"].record_stream(st["post"])
        st["send"].record_stream(st["post"])

    def gemm_reduce_scatter_fp8(self, xq, xs, w_t, w_s, chunks: int) -> torch.Tensor:
        """xq [S, K] fp8 (all rows of the group, this rank's K shard), xs [S, 1], w_t [K, N] column-major,
        w_s [1, N] -> this rank's [S / Wt, N] bf16, partial sums relayed through the other group."""
        cur = torch.cuda.current_stream()
        call = self.w.begin()
        st = self._rs_begin(xq, xs, w_t, w_s, chunks, call, cur)
        for c in range(chunks):
            self._rs_reduce_chunk(st, c)
        self._rs_end(st, cur)
        call.end()
        calls["gemm_reduce_scatter"] += 1
        return st["out"]

    def gemm_rs_then_ag_fp8(self, xq, xs, w_t, w_s, then, w2_t, w2_s, chunks: int):
        """A relayed GEMM + reduce-scatter pipelined into the next relayed all-gather + GEMM, piece by piece:
        each reduced piece of this rank's rows goes through `then(r0, rows_bf16) -> next input rows (bf16)` (any
        row-wise work, e.g. residual add + norm + modulation; it may keep state of its own), is quantized row-wise
        to FP8 and sent while later pieces still land. Returns (this rank's reduced rows [rows, N], the next GEMM's
        output [Wt * rows, N2]). One transport call carries both ops' sends."""
        from xfuser.model_executor.layers.tensor_parallel import _rowwise_fp8

        # both ops' sends in one call: RS C * (1 + Wt), AG C * (1 + Wt) (its scales travel with the pieces)
        if 2 * chunks * (1 + self.Wt) > self.T._TRIGGERS:
            raise ValueError(f"{chunks} pieces exceed the transport's {self.T._TRIGGERS} sends per call")
        cur = torch.cuda.current_stream()
        S = xq.shape[0]
        rows, K2 = S // self.Wt, w2_t.shape[0]
        call = self.w.begin()
        rs = self._rs_begin(xq, xs, w_t, w_s, chunks, call, cur)
        ag = self._ag_begin(rows, K2, chunks, call, cur, chunk_scales=True)
        send = ag["send_buf"] = torch.empty(rows * K2 + rows * 4, dtype=torch.uint8, device=xq.device)
        nq = send[: rows * K2].view(torch.float8_e4m3fn).view(rows, K2)
        ns = send[rows * K2 :].view(torch.float32)
        for c in range(chunks):
            r0, nr = rs["bounds"][c]
            if isinstance(then, RowEpilogue):  # one kernel: reduce + bias + residual + norm + modulation + quantize
                self._rs_reduce_norm_quant_chunk(rs, c, then, nq, ns)
            else:
                q, sc = _rowwise_fp8(then(r0, self._rs_reduce_chunk(rs, c)))  # as the FP8 TP sites quantize
                nq[r0 : r0 + nr] = q
                ns[r0 : r0 + nr] = sc.view(-1)
            self._ag_send_chunk(ag, c, nq, cur)
            self._ag_forward_chunk(ag, c)
        out2 = self._ag_gemm(ag, w2_t, w2_s, cur)
        self._rs_end(rs, cur)
        cur.wait_stream(ag["side"])
        call.end()
        send.record_stream(ag["side"])
        calls["gemm_reduce_scatter"] += 1
        calls["all_gather_gemm"] += 1
        calls["rs_then_ag"] += 1
        return rs["out"], out2


def relay_for(group_name: str) -> Optional[TPRelay]:
    """The relay of TP group `group_name` (collective over the world on first use), or None if it does not apply."""
    if not enabled():
        return None
    r = _relays.get(group_name)
    if r is None:
        try:
            r = TPRelay(group_name)
        except ValueError as err:
            logger.warning(f"TP relay off: {err}")
            r = False
        _relays[group_name] = r
    return r or None
