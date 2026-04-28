"""
Experiment → TrainConfig mapping.

Imports are intentionally lazy so that missing optional dependencies
(e.g. franka_env) do not prevent UR5e-only setups from running.
"""


def _lazy_import(module_path: str, attr: str):
    """Import `attr` from `module_path` only when called."""
    def _get():
        import importlib
        mod = importlib.import_module(module_path)
        return getattr(mod, attr)
    return _get


_CONFIG_FACTORIES = {
    "ram_insertion":             _lazy_import("experiments.ram_insertion.config",             "TrainConfig"),
    "usb_pickup_insertion":      _lazy_import("experiments.usb_pickup_insertion.config",      "TrainConfig"),
    "object_handover":           _lazy_import("experiments.object_handover.config",           "TrainConfig"),
    "egg_flip":                  _lazy_import("experiments.egg_flip.config",                  "TrainConfig"),
    "ur5e_usb_pickup_insertion": _lazy_import("experiments.ur5e_usb_pickup_insertion.config", "TrainConfig"),
}


class _LazyConfigMapping(dict):
    """Dict-like object that resolves lazy factories on first access."""

    def __getitem__(self, key):
        value = super().__getitem__(key)
        if callable(value) and not isinstance(value, type):
            # Factory not yet resolved — call it now
            resolved = value()
            super().__setitem__(key, resolved)
            return resolved
        return value

    def __contains__(self, key):
        return super().__contains__(key)

    def keys(self):
        return super().keys()

    def items(self):
        for k in self.keys():
            yield k, self[k]

    def values(self):
        for k in self.keys():
            yield self[k]


CONFIG_MAPPING = _LazyConfigMapping(_CONFIG_FACTORIES)
