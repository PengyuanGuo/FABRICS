#!/usr/bin/env python3
"""Build a fabrics-flavoured UR5 + Tesollo DG5F-S (left) URDF.

Variant A (cspace fabric) needs:
  - the same 26 revolute joints as the sim URDF
  - joint limits baked to match RobotCfg.joint_pos_limit_overrides
  - empty fixed frames for body-collision spheres
  - no meshes (urdfpy / Warp only need kinematics + fixed frames)

Palm control-point frames (palm_x / ...) are omitted; add them later for
Variant B / get_palm_pose diagnostics.

Default I/O (paths relative to this file):
  in:  ./ur5_dg5fs_left.urdf
  out: ../ur5_tesollo/ur5_dg5fs_left_fabric.urdf

Usage:
  python build_fabric_urdf.py
  python build_fabric_urdf.py --src /path/to/ur5_dg5fs_left.urdf --dst /path/to/out.urdf
"""

from __future__ import annotations

import argparse
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path

# ---------------------------------------------------------------------------
# Limits: mirror isaacsimenvs/.../simtoolreal_tesollo_env_cfg.py RobotCfg
# joint_pos_limit_overrides. JointLimitRepulsion and ADEPT clamp(q*) both
# read limits from this URDF, so they must match the env.
# ---------------------------------------------------------------------------
JOINT_LIMIT_OVERRIDES: dict[str, tuple[float, float]] = {
    "ur5_shoulder_pan_joint": (-2.9762, 3.3070),
    "ur5_shoulder_lift_joint": (-3.9097, 2.3735),
    "ur5_elbow_joint": (-1.6598, 3.1416),
    "ur5_wrist_1_joint": (-2.4002, 3.8830),
    "ur5_wrist_2_joint": (-1.5586, 4.7246),
    "ur5_wrist_3_joint": (-4.5473, 1.7359),
    # Fingers 2-5: mid/distal flexion positive-only (palm side).
    **{f"left_joint_{f}_{seg}": (0.0, 1.571) for f in range(2, 6) for seg in (3, 4)},
    # Thumb mid/distal: negative half is the opposing direction.
    **{f"left_joint_1_{seg}": (-1.571, 0.0) for seg in (3, 4)},
    # Pinky _2: keep adducting half only.
    "left_joint_5_2": (0.0, 0.646),
}

# Canonical revolute order expected by the fabric / env (arm then hand).
EXPECTED_REVOLUTE_JOINTS: tuple[str, ...] = (
    "ur5_shoulder_pan_joint",
    "ur5_shoulder_lift_joint",
    "ur5_elbow_joint",
    "ur5_wrist_1_joint",
    "ur5_wrist_2_joint",
    "ur5_wrist_3_joint",
) + tuple(f"left_joint_{f}_{k}" for f in range(1, 6) for k in range(1, 5))

# Shoulder and wrists keep the sub-box centers. Upper arm, forearm, and hand
# centers come from fit_hand_spheres.py (clipped mesh); hand radii are the fit + 1 mm. Upper arm
# and forearm are four spheres along the long axis. Every sphere has a
# dedicated fixed frame in its parent link coordinates.
_FIT = json.loads(Path(__file__).with_name("hand_collision_spheres.json").read_text())
_HAND_SPHERES = _FIT["spheres"]
_ARM_LONG_SPHERES = _FIT["arm_spheres"]
if [s["parent"] for s in _ARM_LONG_SPHERES].count("ur5_upper_arm_link") != 4:
    raise ValueError("expected 4 fitted spheres on ur5_upper_arm_link")
if [s["parent"] for s in _ARM_LONG_SPHERES].count("ur5_forearm_link") != 4:
    raise ValueError("expected 4 fitted spheres on ur5_forearm_link")


def _fitted_frames(spheres: list[dict]) -> list[tuple[str, str, tuple[float, float, float]]]:
    return [(s["parent"], s["frame"], tuple(s["center"])) for s in spheres]


SPHERE_FRAMES: list[tuple[str, str, tuple[float, float, float]]] = [
    ("ur5_shoulder_link", "ur5_shoulder_link_sphere1", (0.0, -0.0055, 0.001655)),
    *_fitted_frames(_ARM_LONG_SPHERES),
    ("ur5_wrist_1_link", "ur5_wrist_1_link_sphere1", (0.0, 0.00485, -0.003073)),
    ("ur5_wrist_2_link", "ur5_wrist_2_link_sphere1", (0.0, -0.00485, -0.003725)),
    ("ur5_wrist_3_link", "ur5_wrist_3_link_sphere1", (0.0, -0.00275, -0.017225)),
    *_fitted_frames(_HAND_SPHERES),
]
# Arm radii are hand-tuned in Isaac Sim (not mesh-enclosing): thin tube
# middles, larger joint housings. The fitted JSON supplies only their centers.
ARM_RADII = {
    "ur5_shoulder_link_sphere1": 0.105,
    "ur5_upper_arm_link_sphere1": 0.06,
    "ur5_upper_arm_link_sphere2": 0.06,
    "ur5_upper_arm_link_sphere3": 0.06,
    "ur5_upper_arm_link_sphere4": 0.09,
    "ur5_forearm_link_sphere1": 0.045,
    "ur5_forearm_link_sphere2": 0.04,
    "ur5_forearm_link_sphere3": 0.04,
    "ur5_forearm_link_sphere4": 0.06,
    "ur5_wrist_1_link_sphere1": 0.055,
    "ur5_wrist_2_link_sphere1": 0.06,
    "ur5_wrist_3_link_sphere1": 0.05,
}
_HAND_RADII = {s["frame"]: s["radius"] for s in _HAND_SPHERES}
SPHERE_RADII = [ARM_RADII.get(frame, _HAND_RADII.get(frame)) for _, frame, _ in SPHERE_FRAMES]
if None in SPHERE_RADII:
    raise ValueError("every sphere frame needs a radius")


def sync_params(path: Path) -> None:
    """Update the sphere arrays without reformatting unrelated YAML settings."""
    text = path.read_text()
    for key, values in (("collision_sphere_frames", [s[1] for s in SPHERE_FRAMES]),
                        ("collision_sphere_radii", SPHERE_RADII)):
        pattern = rf"(      {key}:\n)(?:          - [^\n]+\n)+"
        replacement = r"\g<1>" + "".join(f"          - {value}\n" for value in values)
        text, count = re.subn(pattern, replacement, text)
        if count != 1:
            raise ValueError(f"Expected one {key} array in {path}")
    path.write_text(text)


def _default_src() -> Path:
    return Path(__file__).resolve().parent / "ur5_dg5fs_left.urdf"


def _default_dst() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "ur5_tesollo"
        / "ur5_dg5fs_left_fabric.urdf"
    )


def strip_meshes(root: ET.Element) -> int:
    """Remove <visual> and <collision> from every link. Returns count removed."""
    removed = 0
    for link in root.findall("link"):
        for tag in ("visual", "collision"):
            for elem in list(link.findall(tag)):
                link.remove(elem)
                removed += 1
    return removed


def bake_joint_limits(root: ET.Element) -> list[str]:
    """Overwrite <limit lower/upper> for joints in JOINT_LIMIT_OVERRIDES."""
    updated: list[str] = []
    for joint in root.findall("joint"):
        name = joint.get("name")
        if name not in JOINT_LIMIT_OVERRIDES:
            continue
        if joint.get("type") != "revolute":
            raise ValueError(f"limit override for non-revolute joint: {name}")
        lo, hi = JOINT_LIMIT_OVERRIDES[name]
        limit = joint.find("limit")
        if limit is None:
            limit = ET.SubElement(joint, "limit")
            limit.set("effort", "7.5")
            limit.set("velocity", "3.141592653589793")
        limit.set("lower", f"{lo}")
        limit.set("upper", f"{hi}")
        updated.append(name)
    missing = sorted(set(JOINT_LIMIT_OVERRIDES) - set(updated))
    if missing:
        raise ValueError(f"joints in overrides not found in URDF: {missing}")
    return updated


def _xyz_str(xyz: tuple[float, float, float]) -> str:
    return f"{xyz[0]} {xyz[1]} {xyz[2]}"


def add_sphere_frames(root: ET.Element) -> list[str]:
    """Append empty fixed links/joints for body-sphere origins.

    If frame_name == parent_link, the existing link is used as the sphere
    origin (no new link). Otherwise a new empty child link is attached with
    a fixed joint at the given xyz.
    """
    existing_links = {link.get("name") for link in root.findall("link")}
    existing_joints = {j.get("name") for j in root.findall("joint")}
    frame_names: list[str] = []

    for parent, frame_name, xyz in SPHERE_FRAMES:
        if parent not in existing_links:
            raise ValueError(f"sphere parent link missing: {parent}")

        if frame_name == parent:
            # Use the link origin itself as a sphere frame.
            frame_names.append(frame_name)
            continue

        if frame_name in existing_links:
            raise ValueError(f"sphere frame link already exists: {frame_name}")

        link = ET.SubElement(root, "link")
        link.set("name", frame_name)
        existing_links.add(frame_name)

        joint_name = f"{frame_name}_joint"
        if joint_name in existing_joints:
            raise ValueError(f"sphere frame joint already exists: {joint_name}")

        joint = ET.SubElement(root, "joint")
        joint.set("name", joint_name)
        joint.set("type", "fixed")
        origin = ET.SubElement(joint, "origin")
        origin.set("xyz", _xyz_str(xyz))
        origin.set("rpy", "0 0 0")
        ET.SubElement(joint, "parent").set("link", parent)
        ET.SubElement(joint, "child").set("link", frame_name)
        existing_joints.add(joint_name)
        frame_names.append(frame_name)

    return frame_names


def revolute_joint_names(root: ET.Element) -> list[str]:
    return [
        j.get("name")
        for j in root.findall("joint")
        if j.get("type") == "revolute"
    ]


def build(src: Path, dst: Path) -> dict:
    if not src.is_file():
        raise FileNotFoundError(f"source URDF not found: {src}")

    tree = ET.parse(src)
    root = tree.getroot()
    if root.tag != "robot":
        raise ValueError(f"expected <robot> root, got <{root.tag}>")

    # Keep robot name distinct so it is obvious this is the fabric copy.
    root.set("name", "ur5_dg5fs_left_fabric")

    n_mesh = strip_meshes(root)
    # Fixed joint axes are unused; a zero axis makes urdfpy normalize 0/0.
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed" and joint.find("axis") is not None:
            joint.remove(joint.find("axis"))
    baked = bake_joint_limits(root)
    frames = add_sphere_frames(root)

    rev = revolute_joint_names(root)
    if tuple(rev) != EXPECTED_REVOLUTE_JOINTS:
        raise ValueError(
            "revolute joint order mismatch.\n"
            f"  got:      {rev}\n"
            f"  expected: {list(EXPECTED_REVOLUTE_JOINTS)}"
        )

    dst.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(dst, encoding="utf-8", xml_declaration=True)

    return {
        "src": str(src),
        "dst": str(dst),
        "n_visual_collision_removed": n_mesh,
        "n_limits_baked": len(baked),
        "n_revolute": len(rev),
        "n_sphere_frames": len(frames),
        "sphere_frames": frames,
        "baked_joints": baked,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=_default_src(),
        help="input URDF (default: sibling ur5_dg5fs_left.urdf)",
    )
    parser.add_argument(
        "--dst",
        type=Path,
        default=_default_dst(),
        help="output fabric URDF (default: ../ur5_tesollo/ur5_dg5fs_left_fabric.urdf)",
    )
    parser.add_argument(
        "--print-frames",
        action="store_true",
        help="print collision_sphere_frames list for the params yaml",
    )
    parser.add_argument(
        "--params", type=Path, action="append", default=[],
        help="synchronize frame/radius arrays in this YAML (repeat for mirrors)",
    )
    args = parser.parse_args()

    info = build(args.src.resolve(), args.dst.resolve())
    for path in args.params:
        sync_params(path)
    print(f"wrote {info['dst']}")
    print(
        f"  revolute={info['n_revolute']}  "
        f"limits_baked={info['n_limits_baked']}  "
        f"sphere_frames={info['n_sphere_frames']}  "
        f"meshes_stripped={info['n_visual_collision_removed']}"
    )
    if args.print_frames:
        print("collision_sphere_frames:")
        for name in info["sphere_frames"]:
            print(f"  - {name}")


if __name__ == "__main__":
    main()
