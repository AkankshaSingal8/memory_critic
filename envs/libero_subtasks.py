"""
LIBERO subtask detection per task.

Each task defines:
  - SUBGOAL_NAMES_<TASK>: ordered list of subgoal names
  - SUBGOAL_CHECKS_<TASK>: ordered list of predicate functions (env -> bool)

Tasks supported:
  - KITCHEN_SCENE8: "put both moka pots on the stove" — 4 sequential subgoals
  - KITCHEN_SCENE3: "turn on the stove and put the moka pot on it" — 3 sequential subgoals
"""

import numpy as np


# ============================================================================
# KITCHEN_SCENE8 — Put Both Moka Pots on the Stove
# 4 sequential: POT2_GRASPED -> POT2_ON_STOVE -> POT1_GRASPED -> POT1_ON_STOVE
# ============================================================================

def check_pot2_grasped(env):
    return env._check_grasp(
        gripper=env.robots[0].gripper,
        object_geoms=[g for g in env.sim.model.geom_names if g is not None and "moka_pot_2" in g],
    )


def check_pot2_on_stove(env):
    return env._eval_predicate(("on", "moka_pot_2", "flat_stove_1_cook_region"))


def check_pot1_grasped(env):
    return env._check_grasp(
        gripper=env.robots[0].gripper,
        object_geoms=[g for g in env.sim.model.geom_names if g is not None and "moka_pot_1" in g],
    )


def check_pot1_on_stove(env):
    return env._eval_predicate(("on", "moka_pot_1", "flat_stove_1_cook_region"))


SUBGOAL_NAMES_KITCHEN_SCENE8 = [
    "pot2_grasped",
    "pot2_on_stove",
    "pot1_grasped",
    "pot1_on_stove",
]

SUBGOAL_CHECKS_KITCHEN_SCENE8 = [
    check_pot2_grasped,
    check_pot2_on_stove,
    check_pot1_grasped,
    check_pot1_on_stove,
]


# ============================================================================
# KITCHEN_SCENE3 — Turn on the Stove and Put the Moka Pot on It
# 3 sequential: STOVE_ON -> POT_GRASPED -> POT_ON_STOVE
# (BDDL goal: (And (Turnon flat_stove_1) (On moka_pot_1 flat_stove_1_cook_region)))
# ============================================================================

def check_stove_on_ks3(env):
    """Check whether flat_stove_1 is turned on."""
    return env._eval_predicate(("turnon", "flat_stove_1"))


def check_moka_grasped_ks3(env):
    return env._check_grasp(
        gripper=env.robots[0].gripper,
        object_geoms=[g for g in env.sim.model.geom_names if g is not None and "moka_pot_1" in g],
    )


def check_moka_on_stove_ks3(env):
    return env._eval_predicate(("on", "moka_pot_1", "flat_stove_1_cook_region"))


SUBGOAL_NAMES_KITCHEN_SCENE3 = [
    "stove_on",
    "moka_pot_grasped",
    "moka_pot_on_stove",
]

SUBGOAL_CHECKS_KITCHEN_SCENE3 = [
    check_stove_on_ks3,
    check_moka_grasped_ks3,
    check_moka_on_stove_ks3,
]


# ============================================================================
# KITCHEN_SCENE4 — Put the Black Bowl in the Bottom Drawer of the Cabinet and Close It
# Init: bottom drawer is open. 3 sequential subgoals:
#   BOWL_GRASPED -> BOWL_IN_DRAWER -> DRAWER_CLOSED
# (BDDL goal: (And (Close white_cabinet_1_bottom_region)
#                  (In akita_black_bowl_1 white_cabinet_1_bottom_region)))
# ============================================================================

def check_bowl_grasped_ks4(env):
    return env._check_grasp(
        gripper=env.robots[0].gripper,
        object_geoms=[g for g in env.sim.model.geom_names if g is not None and "akita_black_bowl" in g],
    )


def check_bowl_in_drawer_ks4(env):
    return env._eval_predicate(("in", "akita_black_bowl_1", "white_cabinet_1_bottom_region"))


def check_drawer_closed_ks4(env):
    return env._eval_predicate(("close", "white_cabinet_1_bottom_region"))


SUBGOAL_NAMES_KITCHEN_SCENE4 = [
    "bowl_grasped",
    "bowl_in_drawer",
    "drawer_closed",
]

SUBGOAL_CHECKS_KITCHEN_SCENE4 = [
    check_bowl_grasped_ks4,
    check_bowl_in_drawer_ks4,
    check_drawer_closed_ks4,
]


# ============================================================================
# KITCHEN_SCENE6 — Put the Yellow and White Mug in the Microwave and Close It
# Init: microwave is open. 3 sequential subgoals:
#   MUG_GRASPED -> MUG_IN_MICROWAVE -> MICROWAVE_CLOSED
# (BDDL goal: (And (In white_yellow_mug_1 microwave_1_heating_region)
#                  (Close microwave_1)))
# ============================================================================

def check_mug_grasped_ks6(env):
    return env._check_grasp(
        gripper=env.robots[0].gripper,
        object_geoms=[g for g in env.sim.model.geom_names if g is not None and "white_yellow_mug" in g],
    )


def check_mug_in_microwave_ks6(env):
    return env._eval_predicate(("in", "white_yellow_mug_1", "microwave_1_heating_region"))


def check_microwave_closed_ks6(env):
    return env._eval_predicate(("close", "microwave_1"))


SUBGOAL_NAMES_KITCHEN_SCENE6 = [
    "mug_grasped",
    "mug_in_microwave",
    "microwave_closed",
]

SUBGOAL_CHECKS_KITCHEN_SCENE6 = [
    check_mug_grasped_ks6,
    check_mug_in_microwave_ks6,
    check_microwave_closed_ks6,
]


# ============================================================================
# Backwards-compat aliases (used by older code paths defaulting to KITCHEN_SCENE8)
# ============================================================================

SUBGOAL_NAMES = SUBGOAL_NAMES_KITCHEN_SCENE8
SUBGOAL_CHECKS = SUBGOAL_CHECKS_KITCHEN_SCENE8
NO_OF_SUBTASKS = len(SUBGOAL_NAMES)
