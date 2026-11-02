"""Minimal progress-reporting interface so pipelines don't depend on a particular UI."""

from __future__ import annotations

from typing import Protocol


class Task(Protocol):
    def advance(self, n: int = 1) -> None: ...
    def finish(self, hide: bool = False) -> None: ...


class Reporter(Protocol):
    def task(self, description: str, total: int | None) -> Task: ...
    def log(self, message: str) -> None: ...


class _NullTask:
    def advance(self, n: int = 1) -> None:
        pass

    def finish(self, hide: bool = False) -> None:
        pass


class NullReporter:
    def task(self, description: str, total: int | None) -> Task:
        return _NullTask()

    def log(self, message: str) -> None:
        print(message)


class _RichTask:
    def __init__(self, progress, task_id):
        self._progress, self._id = progress, task_id

    def advance(self, n: int = 1) -> None:
        self._progress.advance(self._id, n)

    def finish(self, hide: bool = False) -> None:
        task = self._progress._tasks[self._id]
        self._progress.update(self._id, total=task.completed, visible=not hide)


class RichReporter:
    """Adapter over a running ``rich.progress.Progress``."""

    def __init__(self, progress):
        self.progress = progress

    def task(self, description: str, total: int | None) -> Task:
        return _RichTask(self.progress, self.progress.add_task(description, total=total))

    def log(self, message: str) -> None:
        self.progress.console.print(message)
