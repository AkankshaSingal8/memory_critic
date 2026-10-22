"""Side-car safety signals for robomimic envs on a controller that publishes none (OSC).

Under safe-robosuite's mink composite controller every env step's info carries
``safety_constraints`` / ``safety_violations`` / ``unsafe_step`` / ``safety_margins``.
Under the dataset's OSC controller safe-robosuite publishes them with EMPTY lists. When enabled, this
side-car runs scripts/safety_pretrain/osc_safety_monitor.py's SideCarSafetyMonitor
(a pure observer: it never writes the live sim's qpos/qvel/ctrl) around every step
and writes the SAME four info keys, so the eval's safety aggregation
(ogpo/utils/eval_safety.py) works unchanged.

Semantics differ from the mink controller by design (see osc_safety_monitor.py):
the monitor measures EXECUTED motion once per policy step (q_des := arm
configuration after the step, q_curr := before it), where the mink controller
flags the REQUESTED motion on every inner sim step.

Enabled through the environment variable below (set by ogpo/main.py from
eval.osc_safety_monitor_config) so that every env make_env builds -- the single
eval env and each AsyncVectorEnv worker process, which inherits os.environ --
gets its own monitor, exactly as ROBOSUITE_CONTROLLER_CONFIG reaches them.
"""

import importlib.util
import os

ENV_VAR = "OGPO_OSC_SAFETY_MONITOR_CONFIG"

_MONITOR_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                             "scripts", "safety_pretrain", "osc_safety_monitor.py")
_monitor_cls = None


def _side_car_class():
    global _monitor_cls
    if _monitor_cls is None:
        spec = importlib.util.spec_from_file_location("osc_safety_monitor", _MONITOR_PATH)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _monitor_cls = mod.SideCarSafetyMonitor
    return _monitor_cls


def configured_monitor_path():
    """The side-car controller JSON from the environment, or None when disabled."""
    path = os.environ.get(ENV_VAR) or None
    if path is not None and not os.path.isfile(path):
        raise FileNotFoundError(f"{ENV_VAR}={path!r} is not a file")
    return path


class SideCarSafetyInfo:
    """Drives one SideCarSafetyMonitor and writes its signals into the step info."""

    def __init__(self, env, controller_json_path):
        self.monitor = _side_car_class()(env, controller_json_path)

    def after_reset(self):
        self.monitor.bind()

    def before_step(self):
        self.monitor.before_step()

    def after_step(self, info):
        # safe-robosuite publishes the four keys with EMPTY lists under a controller
        # without a safety filter (OSC); a non-empty roster means a safe controller is
        # already reporting, and the side-car must not overwrite it.
        existing = info.get('safety_constraints')
        if existing is not None and len(existing) > 0:
            raise RuntimeError(
                "the controller already publishes a safety roster (safe mink controller?); "
                f"the OSC side-car ({ENV_VAR}) must not overwrite them")
        sig = self.monitor.after_step()
        info['safety_constraints'] = list(sig['constraints'])
        info['safety_violations'] = list(sig['violations'])
        info['unsafe_step'] = bool(sig['unsafe_step'])
        info['safety_margins'] = list(sig['margins'])
        return info
