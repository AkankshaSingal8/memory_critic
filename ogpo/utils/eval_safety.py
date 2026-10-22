"""Per-episode safety-violation statistics for the Q-IPM evals.

Pure helpers (no jax, no simulator) used by ``evaluate_qipm`` and
``evaluate_qipm_parallel``. They only engage when the env info carries the
safe-robosuite keys ``safety_constraints`` (constraint names, roster order) and
``safety_violations`` (0/1 per constraint, same order) -- i.e. under the safe
mink composite controller. Under plain OSC safe-robosuite publishes the keys with
EMPTY lists; that is treated as no safety info, nothing here runs, and the
eval's returned dict is unchanged.

A *view* is a named set of constraints; a view's step is violated when ANY of
its constraints is violated on that step. The two-floor monitor config
(configs/safety_monitor/mink_monitor_floor080_slab081.json) uses views to read
both workspace floors off one rollout, e.g.::

    {"floor081": ["joint_limits", "velocity", "collision", "workspace", "keepout"],
     "floor080": ["joint_limits", "velocity", "collision", "workspace"],
     "workspace_floor081": ["workspace", "keepout"]}

Keys returned by :func:`summarize_safety` (all under ``safety/``):
  {c}_step_rate, {c}_episode_rate                 per constraint c
  {c}_min_margin, {c}_mean_margin                 per constraint c, when the info carries
      safety_margins: mean over episodes of the per-episode min / mean of
      info["safety_margins"][c] (the online runner's env/episode_{min,mean}_margin_<c>)
  {v}_step_rate, {v}_episode_rate, {v}_safe_success,
  {v}_first_violation_step (median; NaN if no episode violated v)  per view v
Executed / shield keys -- only when the env publishes envs/shield_diagnostics.py's info keys
(opt-in, OGPO_SHIELD_DIAGNOSTICS=1) on every step of every episode; otherwise none of these
appear and the key set above is unchanged:
  executed_{c}_step_rate, executed_{c}_episode_rate        per constraint c, the same flags
      evaluated at the arm command actually APPLIED (q_safe with the shield on, q_des with it
      off -- so with the shield off they equal the requested {c}_* rates exactly)
  executed_{v}_step_rate, executed_{v}_episode_rate, executed_{v}_safe_success   per view v
  executed_any_step_rate, executed_any_episode_rate, requested_any_step_rate,
  requested_any_episode_rate                               any constraint, executed / requested
  shield_intervention_step_rate, shield_intervention_episode_rate   any filter stage tagged
      in active_constraints held/clipped/reverted the arm (OR over inner sim steps); with the
      shield off this is the would-have-intervened rate
  shield_{s}_step_rate, shield_{s}_episode_rate            per stage s (workspace, keepout,
      collision_revert, infeasible, configuration, other, and qp_deviation -- the QP moved
      the command by > 1e-4 rad; qp_deviation is NOT part of the intervention total)
``*_step_rate`` is the mean over episodes of violated_steps / episode_length;
``*_episode_rate`` the fraction of episodes with >= 1 violated step;
``*_safe_success`` the fraction of episodes that succeeded with zero violated
steps in the view. First-violation steps are 1-indexed (the episode step count
at which the first violation occurred), matching
scripts/safety_pretrain/eval_bc_policy.py.
"""

import json
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

SAFETY_PREFIX = 'safety/'
# Shield stages reported but not counted as an intervention (see envs/shield_diagnostics.py).
NON_INTERVENTION_STAGES = ('qp_deviation',)


def parse_safety_views(views: Any) -> Optional[Dict[str, List[str]]]:
    """Normalise a config value (None, dict, or JSON string) to a dict or None."""
    if views is None:
        return None
    if isinstance(views, str):
        views = views.strip()
        if not views or views.lower() in ('none', 'null'):
            return None
        views = json.loads(views)
    if not isinstance(views, Mapping):
        raise ValueError(f"safety_views must be a mapping view -> [constraints], got {views!r}")
    out = {}
    for name, cons in views.items():
        if isinstance(cons, str) or not isinstance(cons, Sequence):
            raise ValueError(f"safety_views[{name!r}] must be a list of constraint names, got {cons!r}")
        out[str(name)] = [str(c) for c in cons]
    return out


def resolve_views(views: Optional[Mapping[str, Sequence[str]]],
                  constraint_names: Sequence[str]) -> Dict[str, List[int]]:
    """Map each view to its constraint indices in the roster.

    None (or empty) -> {"any": all constraints}. A view naming a constraint that is
    not in the roster raises: silently dropping it would e.g. make a floor-0.81 view
    equal the floor-0.80 one under a config without the keep-out slab.
    """
    names = [str(n) for n in constraint_names]
    if not views:
        return {'any': list(range(len(names)))}
    out = {}
    for view, cons in views.items():
        missing = [c for c in cons if c not in names]
        if missing:
            raise ValueError(
                f"safety view {view!r} names constraints {missing} that are not in the "
                f"controller's roster {names}; check ROBOSUITE_CONTROLLER_CONFIG / eval.safety_views")
        out[view] = [names.index(c) for c in cons]
    return out


def _margins_or_none(margins):
    if margins is None:
        return None
    return [float(m) for m in np.asarray(margins, dtype=float).reshape(-1)]


def step_safety_from_info(info: Mapping) -> Optional[Tuple[List[str], List[int], Optional[List[float]]]]:
    """(names, flags, margins|None) from a single-env info dict, or None without safety keys."""
    if 'safety_violations' not in info or 'safety_constraints' not in info:
        return None
    names, flags = info['safety_constraints'], info['safety_violations']
    # safe-robosuite publishes the keys with EMPTY lists under a controller with
    # no safety filter (plain OSC): that is "no safety info", not a 0-constraint roster.
    if names is None or flags is None or len(names) == 0:
        return None
    return ([str(n) for n in names], [int(bool(f)) for f in np.asarray(flags).reshape(-1)],
            _margins_or_none(info.get('safety_margins')))


def step_safety_from_vector_info(infos: Mapping, i: int) -> Optional[Tuple[List[str], List[int]]]:
    """(names, flags) for env ``i`` from batched gymnasium vector-env infos, or None.

    gymnasium's VectorEnv._add_info stores a list-valued info in an object array
    (one list per env) and an ndarray-valued one in an (n_envs, k) array, with a
    boolean ``_key`` mask marking which envs reported it this step.
    """
    if 'safety_violations' not in infos or 'safety_constraints' not in infos:
        return None
    for key in ('_safety_violations', '_safety_constraints'):
        mask = infos.get(key)
        if mask is not None and not bool(mask[i]):
            return None
    names, flags = infos['safety_constraints'][i], infos['safety_violations'][i]
    if names is None or flags is None or len(names) == 0:  # empty = no filter (OSC)
        return None
    margins = None
    if 'safety_margins' in infos:
        mask = infos.get('_safety_margins')
        if mask is None or bool(mask[i]):
            margins = _margins_or_none(infos['safety_margins'][i])
    return ([str(n) for n in names], [int(bool(f)) for f in np.asarray(flags).reshape(-1)],
            margins)


def step_shield_from_info(info: Mapping) -> Optional[Tuple[List[int], List[str], List[int]]]:
    """(executed flags, stage names, stage flags) from a single-env info, or None.

    The keys are written by envs/shield_diagnostics.py (opt-in); absent -> None.
    """
    keys = ('executed_safety_violations', 'shield_stages', 'shield_active_stages')
    if any(info.get(k) is None for k in keys):
        return None
    return ([int(bool(f)) for f in np.asarray(info[keys[0]]).reshape(-1)],
            [str(n) for n in info[keys[1]]],
            [int(bool(f)) for f in np.asarray(info[keys[2]]).reshape(-1)])


def step_shield_from_vector_info(infos: Mapping, i: int) -> Optional[Tuple[List[int], List[str], List[int]]]:
    """step_shield_from_info for env ``i`` of batched gymnasium vector-env infos."""
    keys = ('executed_safety_violations', 'shield_stages', 'shield_active_stages')
    vals = []
    for k in keys:
        if k not in infos:
            return None
        mask = infos.get(f'_{k}')
        if mask is not None and not bool(mask[i]):
            return None
        if infos[k][i] is None:
            return None
        vals.append(infos[k][i])
    return step_shield_from_info(dict(zip(keys, vals)))


class EpisodeSafetyRecorder:
    """Accumulates one episode's per-step violation flags.

    Call :meth:`record` once per env step that carries safety info (the step that
    ends the episode included, nothing after it).
    """

    def __init__(self):
        self.names: Optional[List[str]] = None
        self.n_steps = 0
        self.flags: List[List[int]] = []
        self.margins: List[List[float]] = []
        # Set by record_step_safely on an inconsistent step: the episode is dropped.
        self.invalid = False
        # Executed / shield-stage flags (envs/shield_diagnostics.py), recorded separately by
        # record_shield; a problem there drops only these (shield_invalid), never the above.
        self.exec_flags: List[List[int]] = []
        self.stage_names: Optional[List[str]] = None
        self.stage_flags: List[List[int]] = []
        self.shield_invalid = False

    def record(self, names: Sequence[str], flags: Sequence[int],
               margins: Optional[Sequence[float]] = None) -> None:
        names = [str(n) for n in names]
        if self.names is None:
            self.names = names
        elif names != self.names:
            raise ValueError(f"safety constraint roster changed mid-episode: {self.names} -> {names}")
        if len(flags) != len(names):
            raise ValueError(f"{len(flags)} violation flags for {len(names)} constraints {names}")
        if margins is not None:
            if len(margins) != len(names):
                raise ValueError(f"{len(margins)} margins for {len(names)} constraints {names}")
            if len(self.margins) != self.n_steps:
                raise ValueError("safety_margins present on some steps of an episode but not others")
            self.margins.append([float(m) for m in margins])
        elif self.margins:
            raise ValueError("safety_margins present on some steps of an episode but not others")
        self.n_steps += 1
        self.flags.append([int(bool(f)) for f in flags])

    def record_shield(self, exec_flags: Sequence[int], stage_names: Sequence[str],
                      stage_flags: Sequence[int]) -> None:
        """Executed flags (roster order) + shield stage flags for the step just recorded.

        Call right AFTER :meth:`record` for the same step.
        """
        if self.n_steps != len(self.exec_flags) + 1:
            raise ValueError("record_shield must follow record() for the same step, once")
        if len(exec_flags) != len(self.names or []):
            raise ValueError(f"{len(exec_flags)} executed flags for {len(self.names or [])} constraints")
        stage_names = [str(n) for n in stage_names]
        if self.stage_names is None:
            self.stage_names = stage_names
        elif stage_names != self.stage_names:
            raise ValueError(f"shield stage roster changed mid-episode: {self.stage_names} -> {stage_names}")
        if len(stage_flags) != len(stage_names):
            raise ValueError(f"{len(stage_flags)} stage flags for {len(stage_names)} stages")
        self.exec_flags.append([int(bool(f)) for f in exec_flags])
        self.stage_flags.append([int(bool(f)) for f in stage_flags])

    @property
    def has_data(self) -> bool:
        return self.n_steps > 0 and not self.invalid

    def finish(self, success: float) -> Dict[str, Any]:
        """Per-episode record consumed by :func:`summarize_safety`."""
        k = len(self.names or [])
        flags = np.asarray(self.flags, dtype=np.int64).reshape(self.n_steps, k)
        margins = (np.asarray(self.margins, dtype=np.float64).reshape(self.n_steps, k)
                   if self.margins and len(self.margins) == self.n_steps else None)
        out = {'names': list(self.names or []), 'n_steps': self.n_steps,
               'flags': flags, 'margins': margins, 'success': float(success)}
        # Executed / shield data only when EVERY step carried it.
        if (not self.shield_invalid and self.n_steps > 0
                and len(self.exec_flags) == self.n_steps):
            out['exec_flags'] = np.asarray(self.exec_flags, dtype=np.int64).reshape(self.n_steps, k)
            out['stage_names'] = list(self.stage_names)
            out['stage_flags'] = np.asarray(self.stage_flags, dtype=np.int64).reshape(
                self.n_steps, len(self.stage_names))
        return out


def summarize_safety(episodes: Sequence[Mapping[str, Any]],
                     views: Optional[Mapping[str, Sequence[str]]] = None) -> Dict[str, float]:
    """Aggregate per-episode records (from EpisodeSafetyRecorder.finish) to eval keys.

    Returns {} when no episode carried safety info.
    """
    episodes = [e for e in episodes if e['n_steps'] > 0]
    if not episodes:
        return {}
    names = episodes[0]['names']
    for e in episodes[1:]:
        if e['names'] != names:
            raise ValueError(f"safety constraint roster differs across episodes: {names} vs {e['names']}")
    view_idx = resolve_views(views, names)
    out: Dict[str, float] = {}

    lengths = np.array([e['n_steps'] for e in episodes], dtype=np.float64)
    successes = np.array([float(e['success']) > 0 for e in episodes])

    for c_i, c in enumerate(names):
        counts = np.array([e['flags'][:, c_i].sum() for e in episodes], dtype=np.float64)
        out[f'{SAFETY_PREFIX}{c}_step_rate'] = float(np.mean(counts / lengths))
        out[f'{SAFETY_PREFIX}{c}_episode_rate'] = float(np.mean(counts > 0))

    # Margins: only when every episode carried them (a partial set would make the
    # mean silently cover a different episode population than the rates).
    if all(e.get('margins') is not None for e in episodes):
        for c_i, c in enumerate(names):
            out[f'{SAFETY_PREFIX}{c}_min_margin'] = float(
                np.mean([e['margins'][:, c_i].min() for e in episodes]))
            out[f'{SAFETY_PREFIX}{c}_mean_margin'] = float(
                np.mean([e['margins'][:, c_i].mean() for e in episodes]))

    for v, idx in view_idx.items():
        counts, firsts = [], []
        for e in episodes:
            per_step = (e['flags'][:, idx].sum(axis=1) > 0) if idx else np.zeros(e['n_steps'], bool)
            counts.append(float(per_step.sum()))
            if per_step.any():
                firsts.append(int(np.argmax(per_step)) + 1)  # 1-indexed step count
        counts = np.asarray(counts)
        out[f'{SAFETY_PREFIX}{v}_step_rate'] = float(np.mean(counts / lengths))
        out[f'{SAFETY_PREFIX}{v}_episode_rate'] = float(np.mean(counts > 0))
        out[f'{SAFETY_PREFIX}{v}_safe_success'] = float(np.mean(successes & (counts == 0)))
        # NaN (not omitted) when no episode violated the view, so the key set is
        # stable from one eval to the next -- the same convention as
        # evaluation._aggregate_stats; CsvLogger writes 'nan', wandb accepts it.
        out[f'{SAFETY_PREFIX}{v}_first_violation_step'] = (
            float(np.median(firsts)) if firsts else float('nan'))
    if all(e.get('exec_flags') is not None for e in episodes):
        out.update(_summarize_shield(episodes, names, view_idx, lengths, successes))
    return out


def _rates(per_step_counts, lengths):
    counts = np.asarray(per_step_counts, dtype=np.float64)
    return float(np.mean(counts / lengths)), float(np.mean(counts > 0))


def _summarize_shield(episodes, names, view_idx, lengths, successes) -> Dict[str, float]:
    """Executed-violation and shield-intervention keys (see the module docstring)."""
    out: Dict[str, float] = {}
    p = SAFETY_PREFIX
    for c_i, c in enumerate(names):
        out[f'{p}executed_{c}_step_rate'], out[f'{p}executed_{c}_episode_rate'] = _rates(
            [e['exec_flags'][:, c_i].sum() for e in episodes], lengths)
    for v, idx in view_idx.items():
        counts = [float((e['exec_flags'][:, idx].sum(axis=1) > 0).sum()) if idx else 0.0
                  for e in episodes]
        out[f'{p}executed_{v}_step_rate'], out[f'{p}executed_{v}_episode_rate'] = _rates(
            counts, lengths)
        out[f'{p}executed_{v}_safe_success'] = float(np.mean(successes & (np.asarray(counts) == 0)))
    for tag, key in (('executed', 'exec_flags'), ('requested', 'flags')):
        out[f'{p}{tag}_any_step_rate'], out[f'{p}{tag}_any_episode_rate'] = _rates(
            [float((e[key].sum(axis=1) > 0).sum()) for e in episodes], lengths)

    stages = episodes[0]['stage_names']
    for e in episodes[1:]:
        if e['stage_names'] != stages:
            raise ValueError(f"shield stage roster differs across episodes: {stages} vs {e['stage_names']}")
    for s_i, s in enumerate(stages):
        out[f'{p}shield_{s}_step_rate'], out[f'{p}shield_{s}_episode_rate'] = _rates(
            [e['stage_flags'][:, s_i].sum() for e in episodes], lengths)
    iv = [i for i, s in enumerate(stages) if s not in NON_INTERVENTION_STAGES]
    out[f'{p}shield_intervention_step_rate'], out[f'{p}shield_intervention_episode_rate'] = _rates(
        [float((e['stage_flags'][:, iv].sum(axis=1) > 0).sum()) if iv else 0.0 for e in episodes],
        lengths)
    return out


# ---------------------------------------------------------------------------
# Non-raising entry points for the eval loops. A malformed or inconsistent
# safety info must never kill a (possibly multi-day) run mid-eval: the offending
# episode's safety record is dropped and a warning printed once per message.
# ---------------------------------------------------------------------------

_WARNED = set()


def _warn_once(msg: str) -> None:
    if msg not in _WARNED:
        _WARNED.add(msg)
        print(f"[safety eval] WARNING: {msg}", flush=True)


def record_step_safely(recorder: 'EpisodeSafetyRecorder', extract, *args) -> None:
    """extract(*args) -> (names, flags, margins)|None, then recorder.record; never raises.

    On any error the recorder is invalidated (the episode contributes no safety
    record) and the error is reported once.
    """
    if recorder.invalid:
        return
    try:
        step = extract(*args)
        if step is not None:
            recorder.record(*step)
    except Exception as exc:  # noqa: BLE001 -- eval must keep running
        recorder.invalid = True
        _warn_once(f"dropping an episode's safety record: {type(exc).__name__}: {exc}")


def record_shield_safely(recorder: 'EpisodeSafetyRecorder', extract, *args) -> None:
    """extract(*args) -> (exec flags, stage names, stage flags)|None, then record_shield.

    Never raises. Call right after :func:`record_step_safely` for the same step. Only the
    executed / shield record is dropped on error (``shield_invalid``) -- the requested-
    violation record, and so every pre-existing safety key, is never affected. A step
    with request data but without shield data also drops the episode's shield record
    (finish() requires it on every step).
    """
    if recorder.invalid or recorder.shield_invalid:
        return
    try:
        step = extract(*args)
        if step is not None and recorder.n_steps == len(recorder.exec_flags) + 1:
            recorder.record_shield(*step)
        elif step is not None and recorder.n_steps != len(recorder.exec_flags) + 1:
            raise ValueError("shield data on a step without requested-violation data")
    except Exception as exc:  # noqa: BLE001 -- eval must keep running
        recorder.shield_invalid = True
        _warn_once(f"dropping an episode's executed/shield record: {type(exc).__name__}: {exc}")


def summarize_safety_safely(episodes, views) -> Dict[str, float]:
    """summarize_safety that reports errors (every eval) instead of raising."""
    try:
        return summarize_safety(episodes, views)
    except Exception as exc:  # noqa: BLE001 -- eval must keep running
        print(f"[safety eval] WARNING: no safety/ keys this eval: {type(exc).__name__}: {exc}",
              flush=True)
        return {}
