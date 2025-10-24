"""Tests for the background async runner used by the Flask app."""

from __future__ import annotations

import asyncio

import pytest

from app.runtime import BackgroundAsyncRunner


async def _capture_loop(loop_ids: list[int]) -> int:
    loop_ids.append(id(asyncio.get_running_loop()))
    await asyncio.sleep(0)
    return len(loop_ids)


def test_background_async_runner_reuses_single_loop() -> None:
    runner = BackgroundAsyncRunner()
    loop_ids: list[int] = []
    try:
        first_count = runner.run(_capture_loop(loop_ids))
        second_count = runner.run(_capture_loop(loop_ids))

        assert first_count == 1
        assert second_count == 2
        assert loop_ids[0] == loop_ids[1]
    finally:
        runner.close()


def test_background_async_runner_rejects_calls_after_close() -> None:
    runner = BackgroundAsyncRunner()
    runner.close()
    coro = _capture_loop([])
    with pytest.raises(RuntimeError):
        runner.run(coro)
    coro.close()
    runner.close()
