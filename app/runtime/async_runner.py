"""Background asyncio runner for bridging sync Flask code with async services."""

from __future__ import annotations

import asyncio
import threading
from typing import Awaitable, TypeVar


T = TypeVar("T")


class BackgroundAsyncRunner:
    """Manage a dedicated asyncio loop running on a background thread."""

    def __init__(self, *, name: str = "semantic-cache-loop") -> None:
        self._loop = asyncio.new_event_loop()
        self._thread = threading.Thread(
            target=self._run_loop,
            name=name,
            daemon=True,
        )
        self._ready = threading.Event()
        self._closed = threading.Event()
        self._thread.start()
        self._ready.wait()

    def run(self, coro: Awaitable[T]) -> T:
        """Synchronously execute *coro* on the managed event loop."""
        if self._closed.is_set():
            raise RuntimeError("Async runner has been closed.")

        # Re-enable app loggers before running coroutine (they get disabled during request handling)
        import logging as _logging
        for logger_name in ("app", "app.semantic_cache", "app.semantic_cache.repository", "app.cache", "app.proxy", "app.policy"):
            _logging.getLogger(logger_name).disabled = False

        future = asyncio.run_coroutine_threadsafe(coro, self._loop)
        return future.result()

    def close(self) -> None:
        """Stop the managed loop and release resources."""
        if self._closed.is_set():
            return
        self._closed.set()
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._ready.set()
        try:
            self._loop.run_forever()
        finally:
            pending = [task for task in asyncio.all_tasks(self._loop) if not task.done()]
            for task in pending:
                task.cancel()
            if pending:
                self._loop.run_until_complete(
                    asyncio.gather(*pending, return_exceptions=True)
                )
            self._loop.close()
