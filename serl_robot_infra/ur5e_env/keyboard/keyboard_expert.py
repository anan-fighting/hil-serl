"""
Keyboard-based teleoperation expert for HIL-SERL.

Replaces SpaceMouse with keyboard input.  The class runs a background thread
that tracks which keys are currently held down and exposes a `get_action()`
method with the same signature as SpaceMouseExpert.

Key mapping (default):
  Translation XYZ:
    W / S  →  +X / -X   (forward / backward)
    A / D  →  +Y / -Y   (left / right)
    Q / E  →  +Z / -Z   (up / down)

  Rotation (euler rates):
    I / K  →  +Rx / -Rx  (roll)
    J / L  →  +Ry / -Ry  (pitch)
    U / O  →  +Rz / -Rz  (yaw)

  Gripper:
    F      →  hold-to-close gripper   (button[0] = 1)
    G      →  hold-to-open  gripper   (button[1] = 1)

  Emergency stop:
    ESC    →  set terminate flag (read by UR5eEnv)

The action magnitude per key-press is controlled by `linear_speed` and
`angular_speed` constructor arguments.
"""

import threading
import numpy as np
from typing import Tuple

from pynput import keyboard as pynput_keyboard


class KeyboardExpert:
    """
    Keyboard-based teleoperation interface.

    Usage::

        expert = KeyboardExpert()
        action, buttons = expert.get_action()
        # action: np.ndarray shape (6,) – [dx, dy, dz, drx, dry, drz]
        # buttons: list[int] – [close_gripper, open_gripper]
    """

    # -----------------------------------------------------------------------
    # Key → (axis_index, sign) mapping for motion keys
    # -----------------------------------------------------------------------
    _MOTION_MAP = {
        "w": (0, +1),   # +X
        "s": (0, -1),   # -X
        "a": (1, +1),   # +Y
        "d": (1, -1),   # -Y
        "q": (2, +1),   # +Z
        "e": (2, -1),   # -Z
        "i": (3, +1),   # +Rx
        "k": (3, -1),   # -Rx
        "j": (4, +1),   # +Ry
        "l": (4, -1),   # -Ry
        "u": (5, +1),   # +Rz
        "o": (5, -1),   # -Rz
    }

    _GRIPPER_CLOSE_KEY = "f"
    _GRIPPER_OPEN_KEY  = "g"

    def __init__(self, linear_speed: float = 1.0, angular_speed: float = 1.0):
        """
        Args:
            linear_speed:  Scale applied to translational axes [0-2].
            angular_speed: Scale applied to rotational axes [3-5].
        """
        self.linear_speed  = linear_speed
        self.angular_speed = angular_speed

        self._lock       = threading.Lock()
        self._held_keys: set = set()
        self.terminate   = False

        # Start non-blocking listener
        self._listener = pynput_keyboard.Listener(
            on_press=self._on_press,
            on_release=self._on_release,
        )
        self._listener.daemon = True
        self._listener.start()

        self._print_help()

    # -----------------------------------------------------------------------
    # pynput callbacks
    # -----------------------------------------------------------------------
    def _key_to_str(self, key) -> str:
        """Normalise a pynput Key / KeyCode to a lowercase string."""
        if hasattr(key, "char") and key.char is not None:
            return key.char.lower()
        return str(key).replace("Key.", "").lower()

    def _on_press(self, key):
        k = self._key_to_str(key)
        if k == "esc":
            self.terminate = True
            return
        with self._lock:
            self._held_keys.add(k)

    def _on_release(self, key):
        k = self._key_to_str(key)
        with self._lock:
            self._held_keys.discard(k)

    # -----------------------------------------------------------------------
    # Public interface (same as SpaceMouseExpert)
    # -----------------------------------------------------------------------
    def get_action(self) -> Tuple[np.ndarray, list]:
        """
        Returns:
            action  – np.ndarray shape (6,):  [dx, dy, dz, drx, dry, drz]
            buttons – list[int] shape (2,):   [close_gripper, open_gripper]
        """
        action = np.zeros(6, dtype=np.float32)

        with self._lock:
            held = set(self._held_keys)

        for key, (idx, sign) in self._MOTION_MAP.items():
            if key in held:
                speed = self.linear_speed if idx < 3 else self.angular_speed
                action[idx] += sign * speed

        close_gripper = int(self._GRIPPER_CLOSE_KEY in held)
        open_gripper  = int(self._GRIPPER_OPEN_KEY  in held)
        buttons = [close_gripper, open_gripper]

        return action, buttons

    def close(self):
        """Stop the background keyboard listener."""
        self._listener.stop()

    # -----------------------------------------------------------------------
    # Help text
    # -----------------------------------------------------------------------
    @staticmethod
    def _print_help():
        print(
            "\n"
            "┌─────────────────────────────────────────┐\n"
            "│         Keyboard Teleoperation           │\n"
            "├──────────────┬──────────────────────────┤\n"
            "│  W / S       │  +X / -X  (forward/back) │\n"
            "│  A / D       │  +Y / -Y  (left / right) │\n"
            "│  Q / E       │  +Z / -Z  (up / down)    │\n"
            "│  I / K       │  +Rx / -Rx (roll)        │\n"
            "│  J / L       │  +Ry / -Ry (pitch)       │\n"
            "│  U / O       │  +Rz / -Rz (yaw)         │\n"
            "│  F (hold)    │  Close gripper            │\n"
            "│  G (hold)    │  Open  gripper            │\n"
            "│  ESC         │  Emergency stop           │\n"
            "└──────────────┴──────────────────────────┘\n"
        )
