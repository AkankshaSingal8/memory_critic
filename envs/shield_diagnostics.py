"""Executed-motion safety flags and shield-intervention stages for the mink controller.

safe-robosuite's ``info['safety_violations']`` / ``['unsafe_step']`` / ``['safety_margins']``
describe the REQUEST: SafetyFilterSolver.solve() evaluates its margins at the raw IK target
``q_des`` before any filtering (Step 0b), so they are identical whether the shield is on or
off. They say nothing about what the arm was actually commanded to do. This module adds,
opt-in, two per-policy-step signals next to them:

* **executed violations** -- the same margin functions evaluated at the arm command that was
  actually APPLIED on each inner sim step:

    q_applied = q_safe  if the controller's apply_intervention is True (shield on)
              = q_des   otherwise (monitor-only: run_controller leaves q_des on the arm)

  computed as ``solver.compute_margins(q_applied)`` -- the solver's own public helper, which
  re-syncs from the live sim (unchanged since the solve: solve() never touches it) and runs
  exactly Step 0b's computation (same reachable-range clamp for the geometric families, raw
  q for joint_limits, ``|q_applied - q_curr| / integration_dt`` vs ``velocity_limits`` for
  velocity) -- then ``solver._detect_constraint_violations`` (same dead-band) for the flags.
  So with the shield OFF executed == requested by construction, bit for bit; that identity is
  the correctness check of this code path. With the shield ON, executed velocity is the rate
  the QP actually planned. Flags OR-reduce and margins min-reduce over the inner steps, like
  the request latch. Caveats: it is the commanded configuration one control step ahead, not
  the post-step measured joint state (the PD tracking error is not included); and q_safe is
  clipped to the joint's physical range, not the buffered band, so executed joint_limits can
  still read slightly negative under the shield.

* **shield stages** -- which filter stage held/clipped/reverted the arm on any inner step,
  from ``last_diagnostics['active_constraints']``, mapped onto the fixed roster
  :data:`SHIELD_STAGES` (``workspace_<name>`` -> workspace, ``keepout_<name>`` -> keepout,
  ``collision_revert`` and its ``collision_<pair>`` detail -> collision_revert, unknown tags ->
  other). Recorded whatever apply_intervention is: with the shield off it is the
  would-have-intervened record. NOTE ``active_constraints`` does not tag the QP's own
  velocity / joint-limit / collision-limit projection, so the extra pseudo-stage
  ``qp_deviation`` (``max|q_des - q_safe| > QP_DEVIATION_TOL`` on any inner step) is reported
  alongside; it is NOT part of the intervention total.

Enabled through the environment variable :data:`ENV_VAR` so every env make_env builds,
AsyncVectorEnv workers included (they inherit os.environ), gets it. Only list-valued info
keys are written, so evaluation._aggregate_stats drops them and no scalar eval key changes:

  info['executed_safety_violations']  0/1 per constraint, info['safety_constraints'] order
  info['executed_safety_margins']     min over inner steps, same order
  info['shield_stages']               SHIELD_STAGES (the fixed roster)
  info['shield_active_stages']        0/1 per stage, OR over the inner steps

The hook is (re)installed lazily before every env step: reset_from_xml_string rebuilds the
robot, its composite controller and a fresh SafetyFilterSolver, so a hook on the old object
would silently stop firing.
"""

import os
import weakref

import numpy as np

ENV_VAR = "OGPO_SHIELD_DIAGNOSTICS"
SHIELD_STAGES = ("workspace", "keepout", "collision_revert", "infeasible", "configuration",
                 "other", "qp_deviation")
# Stages that count as a shield intervention (everything tagged in active_constraints).
INTERVENTION_STAGES = SHIELD_STAGES[:-1]
QP_DEVIATION_TOL = 1e-4  # rad
_HOOK_ATTR = "_ogpo_shield_diag_log"


def enabled():
    return os.environ.get(ENV_VAR, "").strip().lower() in ("1", "true", "yes", "on")


def canonical_stage(tag):
    """Map one ``active_constraints`` tag onto SHIELD_STAGES."""
    tag = str(tag)
    if tag.startswith("workspace"):
        return "workspace"
    if tag.startswith("keepout"):
        return "keepout"
    if tag.startswith("collision"):  # collision_revert + its collision_<pair> detail
        return "collision_revert"
    if tag in ("infeasible", "configuration"):
        return tag
    return "other"


def _composite_controller(base_env):
    robots = getattr(base_env, "robots", None) or []
    if not robots:
        return None
    return getattr(robots[0], "composite_controller", None)


def _install(cc, solver):
    """Wrap solver.solve to log (stages, executed margins, executed flags) per inner step.

    The wrapper is stored ON the solver, so it must not hold the solver or its controller
    strongly: a bound method or a closure cell would close a reference cycle, and every reset
    (which builds a fresh robot/controller/solver) would then leave the old ones -- MuJoCo model
    and data copies included -- to the cyclic GC, which rarely runs: ~1 GB per few dozen episodes
    across the eval workers, OOM-killing a worker mid-eval (measured 2026-10-14).
    """
    log = []
    inner = type(solver).solve  # unbound: no reference to this solver
    solver_ref, cc_ref = weakref.ref(solver), weakref.ref(cc)

    def solve(q_des, q_curr=None):
        solver, cc = solver_ref(), cc_ref()
        q_safe = inner(solver, q_des, q_curr)
        d = solver.last_diagnostics
        if q_curr is not None:  # never taken by run_controller (it filters from the live sim)
            raise NotImplementedError("shield diagnostics assume solve(q_des) with live-sim q_curr")
        applied = q_safe if cc.apply_intervention else np.asarray(q_des, dtype=float)
        # compute_margins leaves the mink configuration at q_curr; solve() left it at q_next.
        # Nothing reads it before the next _sync_state, but restore it anyway.
        post = solver.configuration.data.qpos.copy()
        try:
            margins = solver.compute_margins(applied)
        finally:
            solver._set_configuration(post)
        flags = solver._detect_constraint_violations(margins)
        stages = {canonical_stage(t) for t in d["active_constraints"]}
        if float(np.max(np.abs(np.asarray(d["q_error"], dtype=float)))) > QP_DEVIATION_TOL:
            stages.add("qp_deviation")
        log.append((stages, margins, flags))
        return q_safe

    solver.solve = solve
    setattr(solver, _HOOK_ATTR, log)
    return log


class ShieldDiagnostics:
    """Per-wrapper driver: before_step() (re)hooks and clears, after_step(info) publishes."""

    def __init__(self):
        self._log = None
        self._cc = None

    def before_step(self, base_env):
        cc = _composite_controller(base_env)
        solver = getattr(cc, "safety_filter_solver", None) if cc is not None else None
        self._cc = cc if solver is not None else None
        if solver is None:
            self._log = None
            return
        log = getattr(solver, _HOOK_ATTR, None)
        if log is None:
            log = _install(cc, solver)
        log.clear()
        self._log = log

    def after_step(self, info):
        if self._log is None:
            return info
        names = list(info.get("safety_constraints") or [])
        if not self._log:
            if getattr(self._cc, "_arm_target_qpos", None) is not None:
                raise RuntimeError("shield diagnostics: the solver hook did not fire this step")
            return info
        solver_names = list(self._log[0][1])
        if names != solver_names:
            raise RuntimeError(f"shield diagnostics: info roster {names} != solver {solver_names}")
        flags = [int(any(bool(f[n]) for _, _, f in self._log)) for n in names]
        margins = [float(min(m[n] for _, m, _ in self._log)) for n in names]
        hit = set().union(*(s for s, _, _ in self._log))
        info["executed_safety_violations"] = flags
        info["executed_safety_margins"] = margins
        info["shield_stages"] = list(SHIELD_STAGES)
        info["shield_active_stages"] = [int(s in hit) for s in SHIELD_STAGES]
        return info
