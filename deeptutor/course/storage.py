"""Atomic, per-course JSON persistence under data/user/workspace/course/."""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
import shutil
from typing import Any, TypeVar

from deeptutor.services.file_io import atomic_write_text
from deeptutor.services.path_service import get_path_service

from .models import Course, CourseClass, CourseInputs, CourseOutline, CourseProgress

T = TypeVar("T")


def _read(path: Path, model: type[T]) -> T | None:
    try:
        if path.exists():
            return model.model_validate(json.loads(path.read_text(encoding="utf-8")))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return None


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    atomic_write_text(path, json.dumps(value.model_dump(mode="json"), ensure_ascii=False, indent=2))


class CourseStorage:
    def __init__(self) -> None:
        self._lock = asyncio.Lock()

    @property
    def root(self) -> Path:
        return get_path_service().get_workspace_dir() / "course"

    def course_root(self, course_id: str) -> Path:
        return self.root / f"course_{course_id}"

    def _path(self, course_id: str, name: str) -> Path:
        return self.course_root(course_id) / name

    def ensure(self, course_id: str) -> Path:
        root = self.course_root(course_id)
        (root / "classes").mkdir(parents=True, exist_ok=True)
        return root

    def list_ids(self) -> list[str]:
        if not self.root.exists():
            return []
        return [p.name[7:] for p in self.root.iterdir() if p.is_dir() and p.name.startswith("course_")]

    def save_course(self, value: Course) -> None:
        self.ensure(value.id)
        _write(self._path(value.id, "manifest.json"), value)

    def load_course(self, course_id: str) -> Course | None:
        return _read(self._path(course_id, "manifest.json"), Course)

    def save_inputs(self, course_id: str, value: CourseInputs) -> None:
        self.ensure(course_id)
        _write(self._path(course_id, "inputs.json"), value)

    def load_inputs(self, course_id: str) -> CourseInputs | None:
        return _read(self._path(course_id, "inputs.json"), CourseInputs)

    def save_outline(self, value: CourseOutline) -> None:
        self.ensure(value.course_id)
        _write(self._path(value.course_id, "outline.json"), value)

    def load_outline(self, course_id: str) -> CourseOutline | None:
        return _read(self._path(course_id, "outline.json"), CourseOutline)

    def clear_classes(self, course_id: str) -> None:
        directory = self._path(course_id, "classes")
        if directory.exists():
            shutil.rmtree(directory, ignore_errors=True)
        directory.mkdir(parents=True, exist_ok=True)

    def delete_outline(self, course_id: str) -> None:
        self._path(course_id, "outline.json").unlink(missing_ok=True)

    def save_progress(self, value: CourseProgress) -> None:
        self.ensure(value.course_id)
        _write(self._path(value.course_id, "progress.json"), value)

    def load_progress(self, course_id: str) -> CourseProgress | None:
        return _read(self._path(course_id, "progress.json"), CourseProgress)

    def save_class(self, value: CourseClass) -> None:
        self.ensure(value.course_id)
        _write(self._path(value.course_id, f"classes/{value.id}.json"), value)

    def load_class(self, course_id: str, class_id: str) -> CourseClass | None:
        return _read(self._path(course_id, f"classes/{class_id}.json"), CourseClass)

    def list_classes(self, course_id: str) -> list[CourseClass]:
        directory = self._path(course_id, "classes")
        if not directory.exists():
            return []
        items = [_read(path, CourseClass) for path in directory.glob("*.json")]
        return sorted((item for item in items if item), key=lambda item: (item.order, item.created_at))

    def append_log(self, course_id: str, message: str) -> None:
        path = self._path(course_id, "log.md")
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(f"- {message.strip()}\n")

    def delete(self, course_id: str) -> bool:
        root = self.course_root(course_id)
        if not root.exists():
            return False
        shutil.rmtree(root, ignore_errors=True)
        return not root.exists()


_storage: CourseStorage | None = None


def get_course_storage() -> CourseStorage:
    global _storage
    if _storage is None:
        _storage = CourseStorage()
    return _storage
