"""配置加载：YAML/JSON + 环境变量覆盖 + 必填校验 + 配置 hash。"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from .errors import ConfigError


def _canonical(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str)


def deep_merge(base: dict[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    """递归合并配置（override 优先，返回新字典）。"""
    out = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[key] = deep_merge(dict(out[key]), value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def _apply_env_overrides(data: dict[str, Any], prefix: str = "CB") -> dict[str, Any]:
    """环境变量覆盖：`CB_<SECTION>__<KEY>=value`，value 按 YAML 解析。

    示例：`CB_RUNTIME__NUM_ENVS=32`、`CB_PATHS__RUNS_DIR=/tmp/runs`。
    """
    out = copy.deepcopy(data)
    for raw_key, raw_value in os.environ.items():
        if not raw_key.startswith(prefix + "_"):
            continue
        path = raw_key[len(prefix) + 1 :].lower().split("__")
        try:
            value = yaml.safe_load(raw_value)
        except yaml.YAMLError as exc:  # pragma: no cover - 非法环境变量
            raise ConfigError(f"invalid env override {raw_key}={raw_value!r}", error=str(exc)) from exc
        cursor = out
        for part in path[:-1]:
            node = cursor.get(part)
            if not isinstance(node, dict):
                node = {}
                cursor[part] = node
            cursor = node
        cursor[path[-1]] = value
    return out


@dataclass
class Config:
    """强类型访问包装：属性式读取 + 必填校验 + 稳定 hash。"""

    data: dict[str, Any]
    path: Path | None = None
    _hash: str | None = field(default=None, repr=False)

    def get(self, key_path: str, default: Any = None) -> Any:
        """按 `a.b.c` 路径读取；缺失返回 default。"""
        cursor: Any = self.data
        for part in key_path.split("."):
            if not isinstance(cursor, Mapping) or part not in cursor:
                return default
            cursor = cursor[part]
        return cursor

    def require(self, key_paths: str | Iterable[str]) -> None:
        """断言若干路径存在且非 None，否则 fail-fast。"""
        if isinstance(key_paths, str):
            key_paths = [key_paths]
        missing = [key for key in key_paths if self.get(key) is None]
        if missing:
            raise ConfigError("missing required config keys", keys=sorted(missing), path=str(self.path))

    def section(self, name: str) -> Config:
        """返回子配置视图。"""
        value = self.get(name)
        if not isinstance(value, Mapping):
            raise ConfigError(f"config section {name!r} is missing or not a mapping", path=str(self.path))
        return Config(dict(value), self.path)

    def with_overrides(self, overrides: Mapping[str, Any]) -> Config:
        """返回合并后的新配置。"""
        return Config(deep_merge(self.data, overrides), self.path)

    def hash(self) -> str:
        """配置内容 hash（进入 checkpoint / 实验记录）。"""
        if self._hash is None:
            self._hash = hashlib.sha256(_canonical(self.data).encode("utf-8")).hexdigest()
        return self._hash

    def to_dict(self) -> dict[str, Any]:
        """返回深拷贝字典。"""
        return copy.deepcopy(self.data)


def load_config(
    path: str | os.PathLike[str],
    overrides: Mapping[str, Any] | None = None,
    *,
    env_prefix: str = "CB",
    required: Iterable[str] | None = None,
) -> Config:
    """加载 YAML/JSON 配置并应用 overrides / 环境变量 / 必填校验。"""
    file_path = Path(path).expanduser()
    if not file_path.is_file():
        raise ConfigError("config file not found", path=str(file_path))
    suffix = file_path.suffix.lower()
    try:
        text = file_path.read_text(encoding="utf-8")
        raw = json.loads(text) if suffix == ".json" else yaml.safe_load(text)
    except (OSError, yaml.YAMLError, json.JSONDecodeError) as exc:
        raise ConfigError("failed to parse config", path=str(file_path), error=str(exc)) from exc
    if not isinstance(raw, Mapping):
        raise ConfigError("config root must be a mapping", path=str(file_path))
    data = dict(raw)
    if overrides:
        data = deep_merge(data, overrides)
    data = _apply_env_overrides(data, prefix=env_prefix)
    cfg = Config(data, file_path)
    if required:
        cfg.require(required)
    return cfg


def resolve_path(cfg: Config, key_path: str, *, must_exist: bool = False) -> Path:
    """解析配置中的路径（支持 `~` 与环境变量），可选存在性校验。"""
    value = cfg.get(key_path)
    if value is None:
        raise ConfigError("path config missing", key=key_path, path=str(cfg.path))
    resolved = Path(os.path.expandvars(os.path.expanduser(str(value)))).resolve()
    if must_exist and not resolved.exists():
        raise ConfigError("configured path does not exist", key=key_path, resolved=str(resolved))
    return resolved
