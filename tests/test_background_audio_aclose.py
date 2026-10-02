from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from livekit.agents.voice.background_audio import _TRACK_NAME, BackgroundAudioPlayer

pytestmark = pytest.mark.unit


class _FakeLocalParticipant:
    def __init__(self, room: _FakeRoom) -> None:
        self._room = room
        self.track_publications: dict[str, SimpleNamespace] = {}
        self.unpublished: list[str] = []

    async def publish_track(self, track: object, options: object) -> SimpleNamespace:
        pub = SimpleNamespace(sid="TR_bg", name=_TRACK_NAME)
        self.track_publications[pub.sid] = pub
        return pub

    async def unpublish_track(self, track_sid: str) -> None:
        self.unpublished.append(track_sid)
        if not self._room.isconnected():
            # after Room.disconnect() nothing routes the FFI reply back: it never returns
            await asyncio.Future()


class _FakeRoom:
    def __init__(self, *, connected: bool) -> None:
        self._connected = connected
        self.local_participant = _FakeLocalParticipant(self)

    def isconnected(self) -> bool:
        return self._connected


async def _started_player(room: _FakeRoom) -> BackgroundAudioPlayer:
    player = BackgroundAudioPlayer()
    await player.start(room=room)  # type: ignore[arg-type]
    return player


async def test_aclose_after_room_disconnect_returns() -> None:
    # a job's shutdown callbacks run after the room disconnected, and the server may have
    # closed the room without the local publication being dropped
    room = _FakeRoom(connected=False)
    player = await _started_player(room)

    await asyncio.wait_for(player.aclose(), timeout=1)

    assert room.local_participant.unpublished == []


async def test_aclose_while_connected_unpublishes() -> None:
    room = _FakeRoom(connected=True)
    player = await _started_player(room)

    await asyncio.wait_for(player.aclose(), timeout=1)

    assert room.local_participant.unpublished == ["TR_bg"]
