"""Mouse-clickable config form (Textual), generated from the :mod:`.config` dataclasses."""

from __future__ import annotations

import dataclasses
import typing
from pathlib import Path
from typing import Any, Literal

from textual.app import App, ComposeResult
from textual.containers import Horizontal, VerticalScroll
from textual.widgets import Button, Collapsible, Footer, Header, Input, Label, Select, Static, Switch

from .config import Config, from_dict, unwrap_optional

# Top-level sections shown per command (everything else is irrelevant to it).
SECTIONS = {
    "viz": ["bags", "topics", "temporal_alignment", "depth", "shim", "viewer", "overlay", "aggregation", "plots"],
    "recon": ["bags", "topics", "temporal_alignment", "depth", "shim", "recon"],
    "config": None,
}


def _wid(path: str) -> str:
    return "f__" + path.replace(".", "__")


def _leaves(obj, prefix: str = ""):
    """Yield ``(dotted_path, type_hint, value, help)`` for every non-dataclass field."""
    hints = typing.get_type_hints(type(obj))
    for f in dataclasses.fields(obj):
        value, hint = getattr(obj, f.name), hints[f.name]
        if dataclasses.is_dataclass(hint):
            yield from _leaves(value, f"{prefix}{f.name}.")
        else:
            yield f"{prefix}{f.name}", hint, value, f.metadata.get("help", "")


def _set_path(d: dict, path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    for p in parents:
        d = d.setdefault(p, {})
    d[leaf] = value


class ConfigForm(App[Config | None]):
    """Edit a :class:`Config`; exits with the edited config on Run, or None on Cancel."""

    CSS = """
    Collapsible { margin: 0 1; }
    .row { height: auto; margin: 0 0 1 0; }
    .name { width: 26; padding: 1 1 0 0; text-style: bold; }
    .widget { width: 1fr; }
    .help { color: $text-muted; padding: 0 0 0 27; }
    #actions { height: auto; dock: bottom; padding: 0 1; background: $panel; }
    #actions Button { margin: 0 1 0 0; }
    #save_path { width: 40; }
    """
    BINDINGS = [("ctrl+r", "run", "Run"), ("ctrl+s", "save", "Save YAML"), ("escape", "cancel", "Cancel")]

    def __init__(self, cfg: Config, mode: Literal["viz", "recon", "config"] = "config", save_path: str = "config.yaml"):
        super().__init__()
        self.cfg, self.mode, self.save_path = cfg, mode, save_path
        self.title = {"viz": "Rerun visualization", "recon": "TSDF reconstruction", "config": "Edit config"}[mode]
        self.sub_title = "click to edit - ctrl+r run - ctrl+s save - esc cancel"
        self._fields: dict[str, tuple[Any, bool]] = {}  # path -> (base type, optional)

    def compose(self) -> ComposeResult:
        yield Header()
        allowed = SECTIONS[self.mode]
        with VerticalScroll():
            groups: dict[str, list] = {}
            for path, hint, value, help_ in _leaves(self.cfg):
                section = path.split(".")[0]
                if allowed is None or section in allowed:
                    groups.setdefault(section if "." in path else "general", []).append((path, hint, value, help_))
            for i, (section, leaves) in enumerate(groups.items()):
                with Collapsible(title=section, collapsed=i > 1):
                    for path, hint, value, help_ in leaves:
                        yield from self._row(path, hint, value, help_)
        with Horizontal(id="actions"):
            if self.mode != "config":
                yield Button("Run", id="run", variant="success")
            yield Button("Save YAML", id="save", variant="primary")
            yield Input(self.save_path, id="save_path", placeholder="path/to/config.yaml")
            yield Button("Cancel", id="cancel", variant="error")
        yield Footer()

    def _row(self, path: str, hint, value, help_: str) -> ComposeResult:
        base, optional = unwrap_optional(hint)
        self._fields[path] = (base, optional)
        wid = _wid(path)
        if base is bool:
            widget = Switch(value=bool(value), id=wid)
        elif typing.get_origin(base) is Literal:
            widget = Select([(str(o), o) for o in typing.get_args(base)], value=value, allow_blank=False, id=wid)
        elif typing.get_origin(base) is list:
            widget = Input(", ".join(str(v) for v in value or []), placeholder="comma separated", id=wid)
        else:
            kind = {int: "integer", float: "number"}.get(base, "text")
            widget = Input("" if value is None else str(value), type=kind, placeholder="(none)" if optional else "", id=wid)
        widget.add_class("widget")
        with Horizontal(classes="row"):
            yield Label(path.split(".")[-1], classes="name")
            yield widget
        if help_:
            yield Static(help_, classes="help")

    def _parse(self, path: str):
        base, optional = self._fields[path]
        widget = self.query_one(f"#{_wid(path)}")
        if isinstance(widget, (Switch, Select)):
            return widget.value
        text = widget.value.strip()
        if typing.get_origin(base) is list:
            (item,) = typing.get_args(base)
            return [item(x.strip()) for x in text.split(",") if x.strip()]
        if not text:
            if optional:
                return None
            if base is str:
                return ""
            raise ValueError(f"{path} is required")
        return base(text) if base in (int, float) else text

    def collect(self) -> Config:
        """Current form values merged over the starting config."""
        data = dataclasses.asdict(self.cfg)
        errors = []
        for path in self._fields:
            try:
                _set_path(data, path, self._parse(path))
            except ValueError as e:
                errors.append(f"{path}: {e}")
        if errors:
            raise ValueError("\n".join(errors))
        return from_dict(Config, data)

    def _collect_or_notify(self) -> Config | None:
        try:
            return self.collect()
        except ValueError as e:
            self.notify(str(e), title="Invalid value", severity="error", timeout=8)
            return None

    def action_run(self) -> None:
        if self.mode == "config" or (cfg := self._collect_or_notify()) is None:
            return
        missing = [b for b in cfg.bags if not Path(b).exists()]
        if missing or not cfg.bags:
            self.notify(f"Bag(s) not found: {missing or '(none given)'}", title="bags", severity="error")
            return
        self.exit(cfg)

    def action_save(self) -> None:
        if (cfg := self._collect_or_notify()) is None:
            return
        path = self.query_one("#save_path", Input).value.strip() or self.save_path
        cfg.save(path)
        self.notify(f"Saved {path}")
        if self.mode == "config":
            self.exit(cfg)

    def action_cancel(self) -> None:
        self.exit(None)

    def on_button_pressed(self, event: Button.Pressed) -> None:
        {"run": self.action_run, "save": self.action_save, "cancel": self.action_cancel}[event.button.id]()


def edit_config(cfg: Config, mode: Literal["viz", "recon", "config"] = "config", save_path: str = "config.yaml") -> Config | None:
    return ConfigForm(cfg, mode, save_path).run()
