"""Pull relay for 2-rank Ulysses groups (XDIT_ULYSSES_SDMA_RELAY=pull).

A 2-rank group exchanges over the one xGMI link between its GPUs. Here part of each direction
goes through a third GPU that only lends memory, as in ROCm/aiter#6236's TP2 all-reduce relay:

- Each GPU's own process allocates one relay pool (XDIT_ULYSSES_SDMA_RELAY_POOL_MB, default 2048)
  for the one sender it serves, at creation, and the sender and its partner map it over IPC.
  Nothing runs on the relay GPU, and no process opens a context on another's GPU (a second
  process's queues there would be time-sliced with its owner's by the hardware scheduler).
- The sender pushes the tail `fraction` of each copy into the bounce region, on its engine for
  the relay GPU, and behind it writes 0 into an HSA signal of the receiver; the rest goes
  straight to the receiver on the engine for the partner, flagged at the receiver. (A GPU-only
  signal, its memory exported to the sender: an IPC signal's dependents are submitted by a ROCr
  host thread, milliseconds late.)
- The receiver queues its pulls up front on its engine for the relay GPU, gated on that signal:
  the SDMA engine itself waits (as for the send triggers), so no kernel spins anywhere (a
  resident spin wave turns a GEMM launched one work-group per CU into two rounds). Behind the
  pulls the engine writes the signal back to 1 and sets the receiver's pull-done word. A wait on
  the compute stream is both words: the direct copies and the pulls.

No other rank takes part in a call, so the pairs need not run the same calls or sizes (CFG x U2,
where the conditional pass carries text tokens the unconditional one does not). Each sender's
relay GPU is the rank of the same pair position in the next pair, so every directed link
carries one leg at most: with pairs (0,1), (2,3): 0->1 via 2, 1->0 via 3, 2->3 via 0, 3->2 via 1.
The world is only consulted once, at creation, for the pairs.

A bounce region (bump-allocated in the pool, at the same offset on both ranks of the pair) is per route lane (the route key without its exchange, plus the signal) and
per call, round robin over _SLOTS: the partner rewrites a region _SLOTS calls later, by when this
rank has pulled it (a rank sends a call's data only after it received the previous call's, pulls
included). Calls are counted per user of the pair (`tag`: Ulysses, FSDP), whose instances keep
their own flags, buffers and bounce, so users on different streams cannot interleave them.

XDIT_ULYSSES_SDMA_RELAY_FRACTION (default 0.375) is the relayed share of each copy, and copies
below XDIT_ULYSSES_SDMA_RELAY_MIN_BYTES (default 491520) go direct. Single node only.
"""

import ctypes
import os
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from xfuser.core.distributed.sdma import transport
from xfuser.logger import init_logger

logger = init_logger(__name__)

_ALIGN = 256
_pairs: Dict[tuple, "PullRelayPair"] = {}


def requested() -> bool:
    from xfuser import envs

    return envs.environment_variables["XDIT_ULYSSES_SDMA_RELAY"]().strip() == "pull"


def applies(process_group: dist.ProcessGroup) -> bool:
    return requested() and process_group.size() == 2 and dist.get_world_size() >= 4


def helpers(pairs: List[Tuple[int, int]]) -> Dict[int, int]:
    """Each rank's relay GPU: the rank at its position in the next pair (pairs sorted)."""
    pairs = sorted(tuple(sorted(p)) for p in pairs)
    out = {}
    for i, pair in enumerate(pairs):
        for k, r in enumerate(pair):
            out[r] = pairs[(i + 1) % len(pairs)][k]
    return out


def split(n: int, fraction: float, min_bytes: int) -> Tuple[int, int]:
    """(direct, relayed) bytes of an n-byte copy: the relayed tail is 256-byte aligned."""
    relayed = int(n * fraction) // _ALIGN * _ALIGN if n >= min_bytes else 0
    return n - relayed, relayed


class _Lane:
    """A bounce region: `mine` where this rank pulls the partner's relayed bytes from, `theirs` where it pushes its
    own (the same offset in the two pools of the pair's relay GPUs)."""

    def __init__(self, mine: int, theirs: int, nbytes: int):
        self.mine, self.theirs, self.nbytes = mine, theirs, nbytes
        self.live = True


class _Route:
    def __init__(self, signal: Optional[int], lane: _Lane, buffers):
        self.signal = signal
        self.lane = lane
        self.buffers = buffers
        self.push: List[tuple] = []
        self.pull: List[int] = []
        self.credit: Optional[int] = None

    @property
    def live(self) -> bool:
        return self.lane.live and all(b.live for b in self.buffers)

    def free(self) -> None:
        lib = transport.load()
        for plan in [x for pair in self.push for x in pair if x is not None] + self.pull:
            lib.sx_plan_free(plan)
        self.push, self.pull = [], []


def _plan(engines, copies, flags) -> int:
    """A raw plan: per entry, an engine, copies (source offset, destination address, bytes), a flag address."""
    lib = transport.load()
    n = len(engines)
    flat = [c for entry in copies for c in entry]
    u64 = ctypes.c_uint64 * max(len(flat), 1)
    pid = lib.sx_plan_create(
        n,
        (ctypes.c_int * n)(*engines),
        (ctypes.c_int * n)(*[len(entry) for entry in copies]),
        u64(*[c[0] for c in flat]),
        u64(*[c[1] for c in flat]),
        u64(*[c[2] for c in flat]),
        None if flags is None else (ctypes.c_uint64 * n)(*flags),
    )
    if pid < 0:
        raise RuntimeError(f"sx_plan_create failed with {pid}")
    return pid


class PullRelayPair:
    """interface.PeerTransport for a 2-rank group whose relayed bytes travel via a lent buffer."""

    host_issued = True
    channels = transport._CHANNELS

    def __init__(self, process_group: dist.ProcessGroup, tag: str = "ulysses", fraction: Optional[float] = None):
        lib = transport.load()
        self.tag = tag
        self.calls = 0
        self.p = transport.group_for(process_group)
        self.process_group = process_group
        self.rank, self.world = self.p.rank, 2
        self.peers = [1 - self.rank]
        ranks = dist.get_process_group_ranks(process_group)
        me, partner = dist.get_rank(), ranks[1 - self.rank]
        every = [None] * dist.get_world_size()
        dist.all_gather_object(every, (me, partner))
        pairs = {tuple(sorted(x)) for x in every}
        if sorted(r for pr in pairs for r in pr) != list(range(dist.get_world_size())):
            raise RuntimeError(f"the SDMA pull relay needs every rank in a 2-rank group; pairs {sorted(pairs)}")
        helper = helpers(list(pairs))
        count = torch.cuda.device_count()
        if dist.get_world_size() > count or self.p.device.index != me % count:
            raise RuntimeError("the SDMA pull relay needs one node with GPU i for rank i")
        self.out_rank, self.in_rank = helper[me], helper[partner]
        self._pools(every, helper, me, partner)
        # Engines of their own per user (a pull engine waits for the partner's data, holding its queue): the first
        # triple for Ulysses, the second for FSDP.
        buses = [lib.sx_bus_of(x % count) for x in (partner, self.out_rank, self.in_rank)] * 2
        engines = (ctypes.c_int * 6)()
        transport._check(
            lib.sx_engine_plan(self.p.device.index, (ctypes.c_int * 6)(*buses), 6, 1, engines), "sx_engine_plan"
        )
        triple = 0 if tag == "ulysses" else 3
        self.e_direct, self.e_push, self.e_pull = list(engines)[triple : triple + 3]
        dev = self.p.device
        self.zero = torch.zeros(1, dtype=torch.int32, device=dev)
        self.one = torch.ones(1, dtype=torch.int32, device=dev)
        # Per signal channel, per call slot: this rank's IPC signal its pulls wait on (the partner's push engine writes
        # 0 into it behind the relayed bytes; this rank's pull engine writes 1 back behind the pulls).
        self._sigs: Dict[int, tuple] = {}
        # Per credit index (FSDP: chunk), per call slot: credits gating this rank's pushes (see credits()).
        self._credits: Dict[int, tuple] = {}
        self.fraction = (
            float(os.environ.get("XDIT_ULYSSES_SDMA_RELAY_FRACTION", "0.375")) if fraction is None else fraction
        )
        self.min_bytes = int(os.environ.get("XDIT_ULYSSES_SDMA_RELAY_MIN_BYTES", str(480 << 10)))
        # [channel][2] words at each rank: word 0 the partner's direct flag, word 1 this rank's pull-done flag.
        self.flags = self.p._buffer(f"pull_relay_{tag}_flags", self.channels * 8, uncached=True)
        self._lanes: Dict[tuple, _Lane] = {}
        self._bump = 0
        self._routes: Dict[tuple, _Route] = {}
        logger.info(
            f"SDMA pull relay ({tag}) for pair {ranks}: rank {me} sends via GPU {self.out_rank}, pulls via GPU {self.in_rank}, "
            f"fraction {self.fraction}, engines direct/push/pull {self.e_direct}/{self.e_push}/{self.e_pull}"
        )

    # interface.PeerTransport --------------------------------------------------------------
    def buffer(self, name: str, nbytes: int) -> transport.SharedBuffer:
        return self.p.buffer(f"pull_relay_{self.tag}_{name}", nbytes)

    def _pools(self, every, helper, me: int, partner: int) -> None:
        """Every rank's relay pool for the one sender it serves, allocated by that rank on its own GPU and mapped by the
        sender and its partner (world-collective, once): no process opens a context on another's GPU, whose queues
        the relay GPU's hardware scheduler would then time-slice with its owner's."""
        lib = transport.load()
        self.pool_bytes = int(os.environ.get("XDIT_ULYSSES_SDMA_RELAY_POOL_MB", "2048")) << 20
        data, flags = ctypes.c_uint64(), ctypes.c_uint64()
        # XDIT_ULYSSES_SDMA_RELAY_POOL_UNCACHED=1: fine-grained (uncached) pool memory, for the bounce traffic other
        # pairs stream through this GPU while it computes.
        uncached = int(os.environ.get("XDIT_ULYSSES_SDMA_RELAY_POOL_UNCACHED", "0") == "1")
        transport._check(lib.sx_malloc(self.pool_bytes, uncached, ctypes.byref(data)), "sx_malloc (relay pool)")
        transport._check(lib.sx_malloc(self.channels * 8, 1, ctypes.byref(flags)), "sx_malloc (relay flags)")
        transport._check(lib.sx_memset(flags.value, 0, self.channels * 8), "sx_memset")
        self._own = (data.value, flags.value)
        size = lib.sx_ipc_handle_size()

        def handle(ptr):
            h = (ctypes.c_uint8 * size)()
            transport._check(lib.sx_ipc_get(ptr, h), "sx_ipc_get")
            return bytes(h)

        handles = [None] * dist.get_world_size()
        dist.all_gather_object(handles, (handle(data.value), handle(flags.value)))

        def open_(h):
            out = ctypes.c_uint64()
            transport._check(lib.sx_ipc_open((ctypes.c_uint8 * size)(*h), ctypes.byref(out)), "sx_ipc_open (relay pool)")
            return out.value

        # out: the pool on this rank's relay GPU (it pushes there, the partner pulls); in: the partner's relay GPU's.
        def mapped(rank):
            return self._own if rank == me else (open_(handles[rank][0]), open_(handles[rank][1]))

        (self.out_data, self.out_flags), (self.in_data, self.in_flags) = mapped(self.out_rank), mapped(self.in_rank)

    def _lane(self, lane, nbytes: int) -> _Lane:
        cur = self._lanes.get(lane)
        if cur is not None and cur.nbytes >= nbytes:
            return cur
        # Bump-allocated in both pools at the same offset: both ranks of the pair build the same routes in order.
        nbytes = -(-max(nbytes, _ALIGN) // _ALIGN) * _ALIGN
        if self._bump + nbytes > self.pool_bytes:
            raise RuntimeError(
                f"SDMA pull relay pool ({self.pool_bytes >> 20} MiB) is full; raise XDIT_ULYSSES_SDMA_RELAY_POOL_MB"
            )
        if cur is not None:
            cur.live = False
        cur = self._lanes[lane] = _Lane(self.in_data + self._bump, self.out_data + self._bump, nbytes)
        self._bump += nbytes
        return cur

    def _exported_signals(self, initial_zero: bool) -> tuple:
        """One GPU-only signal per call slot on this rank, its memory exported, and the partner's attached
        (pair-collective): (ids, local value addresses, partner's value addresses)."""
        lib = transport.load()
        hs = lib.sx_hsa_ipc_handle_size()
        ids, mine, exports = [], [], []
        for _ in range(transport._SLOTS):
            h, nb, off, v = (ctypes.c_uint8 * hs)(), ctypes.c_uint64(), ctypes.c_uint64(), ctypes.c_uint64()
            sid = lib.sx_signal_create_gpu_exported(h, ctypes.byref(nb), ctypes.byref(off), ctypes.byref(v))
            if sid < 0:
                raise RuntimeError(f"sx_signal_create_gpu_exported failed with {sid}")
            ids.append(sid)
            mine.append(v.value)
            exports.append((bytes(h), nb.value, off.value))
            if initial_zero:
                transport.device_bytes(v.value, 8, self.p.device).view(torch.int64).zero_()
        torch.cuda.synchronize()
        every = [None, None]
        dist.all_gather_object(every, exports, group=self.p.process_group)
        theirs = []
        for h, nb, off in every[1 - self.rank]:
            base = ctypes.c_uint64()
            transport._check(lib.sx_hsa_ipc_attach((ctypes.c_uint8 * hs)(*h), nb, ctypes.byref(base)), "sx_hsa_ipc_attach")
            theirs.append(base.value + off)
        return ids, mine, theirs

    def credits(self, index: int) -> tuple:
        """Credit `index`'s signals (pair-collective on first use): per call slot, one per push engine (direct,
        relayed), each starting free (0). A push gated on its credit waits in the engine until the receiver has
        released it (written 0 into the partner's copy, `theirs`), then sets it back to 1 as its engine's last copy.
        ((direct ids, mine, theirs), (relayed ids, mine, theirs))."""
        got = self._credits.get(index)
        if got is None:
            got = self._credits[index] = (self._exported_signals(True), self._exported_signals(True))
        return got

    def _signals(self, channel: int) -> tuple:
        """This rank's pull signals of `channel` (one per call slot) and the addresses of the partner's (pair-collective,
        on the channel's first route). GPU-only signals, as the send triggers, so the SDMA engine polls them (an IPC
        signal's dependents are submitted by a ROCr host thread instead, milliseconds late); each signal's memory is
        exported with hsa_amd_ipc_memory_create so the partner's engine can write it."""
        got = self._sigs.get(channel)
        if got is None:
            got = self._sigs[channel] = self._exported_signals(False)
        return got

    def route(self, key, build, signal, credit: Optional[int] = None, lane=None):
        """interface.PeerTransport.route; `credit` (FSDP): gate this route's pushes on credits(credit); `lane`: the
        bounce lane key, when routes that are never in flight in the same call slot may share one (FSDP: a call's
        inbox region is its slot)."""
        cached = self._routes.get((key, signal))
        if cached is not None and cached.live:
            return cached
        if cached is not None:
            cached.free()
        own = build()[0]
        # The partner's copies to this rank: a route is a function of the sender's rank in the pair.
        saved = (self.rank, self.peers)
        self.rank, self.peers = 1 - saved[0], [saved[0]]
        try:
            theirs = build()[0]
        finally:
            self.rank, self.peers = saved

        def cut(copies):
            out, at = [], 0
            for src, dst, dst_off, n in copies:
                d, r = split(n, self.fraction, self.min_bytes)
                out.append((src, dst, dst_off, d, r, at))
                at += -(-r // _ALIGN) * _ALIGN
            return out, at

        own_cut, own_bytes = cut(own)
        their_cut, their_bytes = cut(theirs)
        per_slot = max(own_bytes, their_bytes, _ALIGN)
        lane_key = lane if lane is not None else (key[1:] if isinstance(key, tuple) and len(key) > 1 else key, signal)
        lane = self._lane(lane_key, transport._SLOTS * per_slot)
        buffers = list({id(c[1]): c[1] for c in own + theirs}.values())
        r = _Route(signal, lane, buffers)
        fl = self.flags
        sig = None if signal is None else self._signals(signal)
        cr = None if credit is None else self.credits(credit)
        r.credit = credit
        one, zero = self.one.data_ptr(), self.zero.data_ptr()
        for slot in range(transport._SLOTS):
            base = slot * per_slot
            direct = [(src, dst.peer_bases[0] + dst_off, d) for src, dst, dst_off, d, _, _ in own_cut]
            relayed = [(src + d, lane.theirs + base + at, n) for src, _, _, d, n, at in own_cut if n]
            # Per engine a data plan (sources relative to the send's tensor) and a tail of absolute 4-byte writes: the
            # direct share flags the partner (1), the relayed share releases the partner's pull signal (0); with a
            # credit, the engine's last copy sets it back to 1 (ROCr gates every copy on it: nothing may follow).
            d_tail = [] if sig is None else [(one, fl.peer_bases[0] + signal * 8, 4)]
            r_tail = [] if sig is None else [(zero, sig[2][slot], 4)]
            if cr is not None:
                d_tail.append((one, cr[0][1][slot], 4))
                r_tail.append((one, cr[1][1][slot], 4))
            r.push.append(
                (
                    _plan([self.e_direct], [direct], None),
                    _plan([self.e_direct], [d_tail], None) if d_tail else None,
                    _plan([self.e_push], [relayed], None),
                    _plan([self.e_push], [r_tail], None) if r_tail else None,
                )
            )
            # Pulls (absolute addresses), each waiting on this rank's signal in the engine; then this rank's pull-done
            # word, and last the signal set back to 1 (ROCr gates every copy on the signal: anything after the re-arm
            # would wait for the next release).
            pulls = [(lane.mine + base + at, dst.ptr + dst_off + d, n) for _, dst, dst_off, d, n, at in their_cut if n]
            if sig is not None:
                pulls += [(self.one.data_ptr(), fl.ptr + signal * 8 + 4, 4), (self.one.data_ptr(), sig[1][slot], 4)]
            r.pull.append(_plan([self.e_pull], [pulls], None))
        self._routes[(key, signal)] = r
        return r

    def begin(self) -> "PullRelayCall":
        index = self.calls % transport._SLOTS
        self.calls += 1
        return PullRelayCall(self, self.p.begin(), index)


class PullRelayCall:
    """interface.Call: a send pushes once the caller's stream reaches it; a signalled send also queues the pulls of the
    partner's sends since the previous signal, which the pull engine starts once the partner releases the signal."""

    def __init__(self, pair: PullRelayPair, call: transport.Call, index: int):
        self.pair, self.call, self.index = pair, call, index
        self.pending: List[_Route] = []

    def _wait(self, words: int, skip: int, stream) -> None:
        c = self.call
        if transport.cp_wait():
            transport._check(
                transport.load().sx_wait_flags_cp(ctypes.c_void_p(stream.cuda_stream), words, 2, skip), "sx_wait_flags_cp"
            )
            c._waits += 1
            return
        transport._check(
            transport.load().sx_wait_flags(
                ctypes.c_void_p(stream.cuda_stream),
                words,
                2,
                skip,
                transport.wait_seconds(),
                c._waits,
                c.id,
                int(transport.on_timeout() == "abort"),
            ),
            "sx_wait_flags",
        )
        c._waits += 1

    def send(self, route: _Route, src: torch.Tensor, stream: torch.cuda.Stream) -> None:
        c, pair, slot = self.call, self.pair, self.index
        if c._sends >= transport._TRIGGERS:
            raise ValueError(f"at most {transport._TRIGGERS} sends per call")
        lib = transport.load()
        d_data, d_tail, r_data, r_tail = route.push[slot]
        credit = None if route.credit is None else pair.credits(route.credit)
        for plan, base, engine in ((d_data, src.data_ptr(), 0), (d_tail, 0, 0), (r_data, src.data_ptr(), 1), (r_tail, 0, 1)):
            if plan is None:
                continue
            if credit is None:
                rc = lib.sx_plan_issue(c.slot, c._sends, plan, base, 0)
            else:
                rc = lib.sx_plan_issue_trigger_dep(c.slot, c._sends, plan, base, 0, credit[engine][0][slot])
            transport._check(rc, "sx_plan_issue")
        transport._check(lib.sx_release(c.slot, c._sends, ctypes.c_void_p(stream.cuda_stream)), "sx_release")
        c._sends += 1
        self.pending.append(route)
        if route.signal is None:
            return
        # The pulls of the partner's sends since the previous signal: queued now, they start in this rank's pull
        # engine once the partner's push engine releases this channel's signal (no kernel waits for them).
        sid = pair._signals(route.signal)[0][slot]
        for r in self.pending:
            transport._check(lib.sx_plan_issue_dep(c.slot, r.pull[slot], 0, pair.one.data_ptr(), sid), "sx_plan_issue_dep")
        self.pending = []

    def wait(self, channel: int, stream: torch.cuda.Stream) -> None:
        # Both words: the partner's direct copies and this rank's pulls of its relayed bytes.
        self._wait(self.pair.flags.ptr + channel * 8, 2, stream)

    def end(self) -> None:
        if self.pending:
            raise RuntimeError("SDMA pull relay call ended with unsignalled sends")
        self.call.end()


def pair_for(process_group: dist.ProcessGroup, tag: str = "ulysses", fraction: Optional[float] = None) -> PullRelayPair:
    """The pull relay of a 2-rank `process_group` for user `tag` (collective over the world on first use)."""
    key = (tuple(dist.get_process_group_ranks(process_group)), tag)
    r = _pairs.get(key)
    if r is None:
        r = _pairs[key] = PullRelayPair(process_group, tag, fraction)
    return r
