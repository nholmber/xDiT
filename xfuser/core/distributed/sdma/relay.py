"""Relayed SDMA transport for 2-rank Ulysses groups (XDIT_ULYSSES_SDMA_RELAY=1).

A 2-rank group (Ulysses-2, e.g. under TP4 x U2 or CFG x U2 on 8 GPUs) exchanges over the one
xGMI link between its two GPUs, while the node's other links idle. This transport implements
interface.PeerTransport for such a group on top of the world group's SDMA transport and splits
every copy to the partner into one slice per other GPU of the node:

- hop 1 (the sender's send): the partner's slice goes straight into its buffer; every other
  slice goes into a staging buffer of another GPU (a helper). All are flagged on channel 2c.
- forward (each helper, on a side stream): once channel 2c has landed from every sender, the
  helper copies each staged slice from its staging into the inbox of that sender's partner,
  flagged on channel 2c + 1. Its own partner gets only the flag, on the engine that carried its
  direct slice, so that flag also lands behind the direct slice.
- receive (`wait(c)`): channel 2c + 1 from every other GPU: the forwarded slices and, behind
  the partner's flag, the direct slice.

This needs every rank of the world to make the same sequence of exchange calls at the same time:
the world is made of 2-rank Ulysses groups running the same model step (TP x U2, CFG x U2); it is
opt-in. The pairs' sizes may differ (CFG x U2 where the conditional pass carries text tokens the
unconditional one does not). So every buffer and route request is agreed on the host, over a gloo
group: a buffer gets the largest size any rank asked for, and when any rank has no route for the
call, every rank rebuilds it and they swap their copies (destination buffer, offset, bytes), so
each helper forwards the other pairs' slices as that pair cut them.

Staging areas are indexed by the world transport's call slot, so a call's staged slices are not
overwritten before the call _SLOTS calls later, by which time they have been forwarded (a sender
sends a call's data only after it received the previous call's, which helpers forwarded after
reading their staging).
"""

from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

import xfuser.envs as envs
from xfuser.core.distributed.sdma import transport
from xfuser.logger import init_logger

logger = init_logger(__name__)

_ALIGN = 256
_relays: Dict[Tuple[int, ...], "RelayPair"] = {}
_side: Dict[torch.device, torch.cuda.Stream] = {}


def requested() -> bool:
    return envs.environment_variables["XDIT_ULYSSES_SDMA_RELAY"]().strip() == "1"


def applies(process_group: dist.ProcessGroup) -> bool:
    """Relay for `process_group`: requested, a 2-rank group, and more GPUs in the world to relay through."""
    return requested() and process_group.size() == 2 and dist.get_world_size() > 2


def _world_group() -> dist.ProcessGroup:
    try:
        from xfuser.core.distributed.parallel_state import get_world_group

        return get_world_group().device_group
    except Exception:  # noqa: BLE001 - plain torch.distributed users (tests)
        return dist.group.WORLD


def _side_stream(device: torch.device) -> torch.cuda.Stream:
    stream = _side.get(device)
    if stream is None:
        stream = _side[device] = torch.cuda.Stream(device=device, priority=-1)
    return stream


def slices(nbytes: int, sender: int, world: int) -> List[Tuple[int, int, int]]:
    """(receiver, offset, bytes) of each non-empty slice of a `sender`'s copy: one per other rank, in
    rank order, 256-byte aligned except the last."""
    others = [x for x in range(world) if x != sender]
    part = (nbytes // len(others)) // _ALIGN * _ALIGN
    out, off = [], 0
    for i, x in enumerate(others):
        size = nbytes - off if i == len(others) - 1 else part
        if size:
            out.append((x, off, size))
        off += size
    return out


class _Route:
    def __init__(self, signal: Optional[int]):
        self.signal = signal
        self.hop1: List[transport.Plan] = []
        self.hop2: List[transport.Plan] = []
        self.stage: Optional[transport.SharedBuffer] = None
        self.buffers: List[transport.SharedBuffer] = []

    @property
    def live(self) -> bool:
        return all(b.live for b in self.buffers) and (self.stage is None or self.stage.live)


class RelayPair:
    """interface.PeerTransport for a 2-rank group, relayed over the world group's SDMA transport."""

    host_issued = True

    def __init__(self, process_group: dist.ProcessGroup):
        self.process_group = process_group
        self.w = transport.group_for(_world_group())
        self.rank, self.world = process_group.rank(), 2
        self.peers = [1 - self.rank]
        self.channels = self.w.channels // 2
        ranks = dist.get_process_group_ranks(process_group)
        mine = (self.w.rank, ranks[1 - self.rank], self.rank)
        every = [None] * self.w.world
        dist.all_gather_object(every, mine, group=self.w.process_group)
        self.partner = {wr: p for wr, p, _ in every}
        self.pair_rank = {wr: r for wr, _, r in every}
        if sorted(self.partner) != list(range(self.w.world)) or any(
            self.partner[self.partner[x]] != x for x in self.partner
        ):
            raise RuntimeError(f"the SDMA relay needs every rank in a 2-rank group; partners {self.partner}")
        self._routes: Dict[tuple, _Route] = {}
        # Host-side agreement on every buffer and route request (see the module docstring).
        self._host = dist.new_group(dist.get_process_group_ranks(self.w.process_group), backend="gloo")
        logger.info(f"SDMA relay for pair {ranks} over {self.w.world} GPUs")

    # interface.PeerTransport --------------------------------------------------------------
    def _every(self, mine):
        every = [None] * self.w.world
        dist.all_gather_object(every, mine, group=self._host)
        return every

    def _max(self, value: int) -> int:
        t = torch.tensor([value], dtype=torch.int64)
        dist.all_reduce(t, op=dist.ReduceOp.MAX, group=self._host)
        return int(t.item())

    def buffer(self, name: str, nbytes: int) -> transport.SharedBuffer:
        # The largest size any rank asks for at this call, so every rank (re)creates the buffer together.
        return self.w.buffer("relay_" + name, self._max(nbytes))

    def _name_of(self, buf: transport.SharedBuffer) -> str:
        for name, b in self.w._buffers.items():
            if b is buf:
                return name
        raise RuntimeError("an SDMA relay route copies into a buffer the world transport does not hold")

    def route(self, key, build, signal):
        cached = self._routes.get((key, signal))
        # Rebuilt by every rank when any rank has to (another pair may have a new shape at this call).
        if not self._max(int(cached is None or not cached.live)):
            return cached
        me, W = self.w.rank, self.w.world
        own = build()[0]
        # Every rank's copies to its partner (each rank builds this route at the same call), so that a helper
        # forwards another pair's slices as that pair cut them, whatever its sizes.
        every = self._every([(self._name_of(dst), dst_off, n) for _, dst, dst_off, n in own])
        theirs = [[(None, self.w._buffers[name], dst_off, n) for name, dst_off, n in copies] for copies in every]
        theirs[me] = own

        def copies_of(sender):
            return theirs[sender]

        # Staging at each helper: [slot][sender] areas of the largest per-sender size.
        def staged_bytes(sender, helper):
            return sum(sz for (_, _, _, n) in copies_of(sender) for (x, _, sz) in slices(n, sender, W) if x == helper)

        per_sender = max(
            [staged_bytes(s, h) for s in range(W) for h in range(W) if h not in (s, self.partner[s])] + [_ALIGN]
        )
        per_sender = -(-per_sender // _ALIGN) * _ALIGN
        r = _Route(signal)
        # One staging buffer per lane (collective: every rank builds the same routes): the route key without its
        # first element (the exchange, which carries the shape), plus the signal. Routes of one call never share a
        # lane, and a new shape grows the lane's buffer instead of adding one (the old one is released and the
        # routes still on it rebuild), so memory stays bounded by the largest route per lane.
        lane = key[1:] if isinstance(key, tuple) and len(key) > 1 else key
        r.stage = self.w.buffer(f"relay_stage_{lane!r}_{signal}", transport._SLOTS * W * per_sender)
        r.buffers = list({id(c[1]): c[1] for rb in theirs for c in rb}.values())

        def stage_offset(slot, sender, j, helper):
            before = sum(
                sz for (_, _, _, n) in copies_of(sender)[:j] for (x, _, sz) in slices(n, sender, W) if x == helper
            )
            return (slot * W + sender) * per_sender + before

        a = None if signal is None else 2 * signal
        b = None if signal is None else 2 * signal + 1
        # A rank whose own route is unchanged rebuilds too: the forwards of another pair may have changed.
        for slot in range(transport._SLOTS):
            for k in ((("relay1", key, signal, slot), a), (("relay2", key, signal, slot), b)):
                old = self.w._routes.pop(k, None)
                if old is not None:
                    old.free()
        for slot in range(transport._SLOTS):

            def hop1(slot=slot):
                out = []
                for q in self.w.peers:
                    mine = []
                    for j, (src, dst, dst_off, n) in enumerate(copies_of(me)):
                        for x, off, sz in slices(n, me, W):
                            if x != q:
                                continue
                            if q == self.partner[me]:
                                mine.append((src + off, dst, dst_off + off, sz))
                            else:
                                mine.append((src + off, r.stage, stage_offset(slot, me, j, q), sz))
                    out.append(mine)
                return out

            def hop2(slot=slot):
                out = []
                for q in self.w.peers:
                    sender = self.partner[q]
                    mine = []
                    if q != self.partner[me]:  # the partner gets only the flag, behind its direct slice
                        for j, (_, dst, dst_off, n) in enumerate(copies_of(sender)):
                            for x, off, sz in slices(n, sender, W):
                                if x == me:
                                    mine.append((stage_offset(slot, sender, j, me), dst, dst_off + off, sz))
                    out.append(mine)
                return out

            r.hop1.append(self.w.route(("relay1", key, signal, slot), hop1, a))
            r.hop2.append(self.w.route(("relay2", key, signal, slot), hop2, b))
        self._routes[(key, signal)] = r
        return r

    def begin(self) -> "RelayCall":
        return RelayCall(self, self.w.begin())


class RelayCall:
    """interface.Call: each send is hop 1 on the caller's stream; a signalled send also queues, on a side
    stream, this rank's forwards of every send since the previous signal, once their hop 1 landed."""

    def __init__(self, relay: RelayPair, call: transport.Call):
        self.relay, self.call = relay, call
        self.pending: List[_Route] = []

    def send(self, route: _Route, src: torch.Tensor, stream: torch.cuda.Stream) -> None:
        slot = self.call.slot
        self.call.send(route.hop1[slot], src, stream)
        self.pending.append(route)
        if route.signal is None:
            return
        side = _side_stream(src.device)
        self.call.wait(2 * route.signal, side)
        for r in self.pending:
            self.call.send(r.hop2[slot], r.stage.local, side)
        self.pending = []

    def wait(self, channel: int, stream: torch.cuda.Stream) -> None:
        self.call.wait(2 * channel + 1, stream)

    def end(self) -> None:
        if self.pending:
            raise RuntimeError("SDMA relay call ended with unsignalled sends")
        self.call.end()


def relay_for(process_group: dist.ProcessGroup) -> RelayPair:
    """The relay transport of a 2-rank `process_group` (collective over the world on first use)."""
    key = tuple(dist.get_process_group_ranks(process_group))
    r = _relays.get(key)
    if r is None:
        r = _relays[key] = RelayPair(process_group)
    return r
