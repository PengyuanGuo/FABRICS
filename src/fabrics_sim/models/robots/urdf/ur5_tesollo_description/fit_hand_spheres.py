#!/usr/bin/env python3
"""Fit hand and long-arm collision spheres to clipped visual meshes (offline, CPU only).

Example, from the project root:
  .venv_isaacsim/bin/python thirdparty/FABRICS/src/fabrics_sim/models/robots/urdf/ur5_tesollo_description/fit_hand_spheres.py \
    --src assets/urdf/ur5_tesollo_description/ur5_dg5fs_left.urdf

The palm is split into a 3-by-2 grid along its two longest axes. Proximal
and middle phalanges are split into two longitudinal sections. Each distal
``left_link_*_4`` mesh is concatenated with its fixed ``*_tip`` child
(expressed in the ``*_4`` frame) and split into three sections, so the
fingertip cap is covered by the same spheres. ``ur5_upper_arm_link`` and
``ur5_forearm_link`` are split into four sections along their longest axis.
Spheres enclose the actual clipped triangles, including cut-edge vertices,
with a 1 mm margin. No simulator or runtime mesh-fitting dependency is
introduced.

``--profile s30`` fits a 30-sphere hand (DextrAH/ADEPT-sized budget; the
Kuka-Allegro model has 22 hand spheres of 31 total) instead of the default 41:
palm 2-by-2 grid (4), fingers 2-5 proximal 2 / middle 1 / distal+tip 2 (5 each),
thumb 2 / 2 / 2 (6). Frames are named ``<link>_s30_<k>`` so both sets live in
one fabric URDF; the arm is not refitted (its 12 spheres are shared)::

  .venv_isaacsim/bin/python .../fit_hand_spheres.py \
    --src assets/urdf/ur5_tesollo_description/ur5_dg5fs_left.urdf --profile s30
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh
import yaml


HAND_LINKS = ("left_link_base",) + tuple(
    f"left_link_{finger}_{segment}" for finger in range(1, 6) for segment in (2, 3, 4)
)
ARM_LINKS = ("ur5_upper_arm_link", "ur5_forearm_link")
ARM_SECTIONS = 4
DISTAL_SECTIONS = 3

# v2 hand (fit on ur5_dg5fs_left_w_mount.urdf, which keeps the knuckle and mount meshes):
# every link from the palm out gets spheres, including the first link left_link_<f>_1;
# the fingertip end may stay up to V2_TIP_ALLOWANCE uncovered.
V2_SECTIONS = {
    "left_link_base": [(0, 1)],  # palm: one enclosing sphere (was a 2-by-2 grid)
    # Thumb: 6 spheres, 2 per phalanx 1_2 / 1_3 / 1_4 (equal radii, V2_THUMB_RADIUS); its
    # rotation link 1_1 sits next to the palm sphere and carries none.
    "left_link_1_2": [(0, 2)], "left_link_1_3": [(0, 2)], "left_link_1_4": [(0, 2)],
    # Index / middle / ring: knuckle 1 (fitted) + 5 equal spheres (V2_FINGER_EQUAL).
    **{f"left_link_{f}_{seg}": [(0, n)] for f in range(2, 5) for seg, n in ((1, 1), (2, 2), (3, 1), (4, 2))},
    # Pinky: cupping link 2, knuckle body 1, middle 1, distal 2.
    "left_link_5_1": [(0, 2)], "left_link_5_2": [(0, 1)], "left_link_5_3": [(0, 1)], "left_link_5_4": [(0, 2)],
}
V2_TIP_ALLOWANCE = 0.0035  # m of the fingertip cap allowed outside its sphere
# Pinky: its single middle-phalanx (_3) sphere takes the radius of its inner distal sphere
# instead of enclosing the whole link.
V2_MIDDLE_MATCH_DISTAL = True
# Index / middle / ring: the 5 spheres after the knuckle (_2 x2, _3, _4 x2) all take the radius
# of the second sphere after the knuckle (left_link_<f>_2_v2_2). No fingertip allowance for them.
V2_FINGER_EQUAL = True
# Thumb: one radius for all 6 spheres; None = median of their fitted radii.
V2_THUMB_RADIUS = None
# Per-sphere overrides, applied last (edited by hand or saved from fabric_viser.py's sphere editor):
#   <frame>: {radius: <m>, offset: [dx, dy, dz]}   offset in metres, in the sphere's parent-link frame
#   <frame>: {delete: true}                        drop the sphere (its frame disappears from the URDF)
V2_OVERRIDES_FILE = Path(__file__).with_name("hand_sphere_overrides_v2.yaml")
# Palm: the V2_PALM_SHRINK[0] largest palm spheres are scaled by V2_PALM_SHRINK[1] (None = keep
# the enclosing radius). Single palm sphere: (1, 0.8) = 20 % smaller than enclosing (58.7 -> 47.0 mm),
# leaving the palm's far corners outside. With the 2-by-2 grid it was (2, 0.9).
V2_PALM_SHRINK = (1, 0.8)
# Palm sphere centre offset [m], added to the fitted centre, in the left_link_base frame:
#   +x = palm (inner) face, the side the fingers flex toward; -x = back of the hand
#   +y = pinky side, -y = index/thumb side
#   +z = from the wrist/mount toward the fingers (palm mesh spans z 0..0.103, fingertips ~0.198)
# Fitted centre for the single palm sphere: (0.0008, -0.0070, 0.0475).
V2_PALM_OFFSET = (-0.003, 0.0, 0.0)

# 30-sphere hand: sections per (finger, segment); palm uses a 2-by-2 grid.
S30_SECTIONS = {
    **{(finger, seg): count for finger in range(2, 6) for seg, count in ((2, 2), (3, 1), (4, 2))},
    **{(1, seg): 2 for seg in (2, 3, 4)},
}


def load_visual_mesh(link: ET.Element, source: Path) -> trimesh.Trimesh:
    meshes = []
    for visual in link.findall("visual"):
        tag = visual.find("geometry/mesh")
        if tag is None:
            continue
        mesh = trimesh.load(source.parent / tag.get("filename"), force="mesh")
        origin = visual.find("origin")
        attrs = {} if origin is None else origin.attrib
        rotation = Rotation.from_euler(
            "xyz", np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")
        ).as_matrix()
        scale = np.fromstring(tag.get("scale", "1 1 1"), sep=" ")
        translation = np.fromstring(attrs.get("xyz", "0 0 0"), sep=" ")
        mesh.vertices = (mesh.vertices * scale) @ rotation.T + translation
        meshes.append(mesh)
    if not meshes:
        raise ValueError(f"{link.get('name')} has no visual mesh")
    return trimesh.util.concatenate(meshes)


def _origin_pose(element: ET.Element | None) -> tuple[np.ndarray, np.ndarray]:
    attrs = {} if element is None else element.attrib
    rotation = Rotation.from_euler("xyz", np.fromstring(attrs.get("rpy", "0 0 0"), sep=" ")).as_matrix()
    translation = np.fromstring(attrs.get("xyz", "0 0 0"), sep=" ")
    return rotation, translation


def distal_with_tip(root: ET.Element, links: dict[str, ET.Element], source: Path, name: str) -> trimesh.Trimesh:
    """``name`` is ``left_link_*_4``. Tip mesh is moved by its fixed-joint origin."""
    tip_name = name[:-2] + "_tip"
    joint = next(
        joint for joint in root.findall("joint")
        if joint.get("type") == "fixed"
        and joint.find("parent").get("link") == name
        and joint.find("child").get("link") == tip_name
    )
    mesh = load_visual_mesh(links[name], source)
    tip = load_visual_mesh(links[tip_name], source)
    rotation, translation = _origin_pose(joint.find("origin"))
    tip.vertices = tip.vertices @ rotation.T + translation
    return trimesh.util.concatenate([mesh, tip])


def section_spheres(name: str, mesh: trimesh.Trimesh, bins: list[tuple[int, int]],
                    frame_tag: str = "") -> list[dict]:
    lo, hi = mesh.bounds
    chunks = [mesh]
    for axis, count in bins:
        sections = []
        for chunk in chunks:
            for index in range(count):
                origin = np.zeros(3)
                normal = np.eye(3)[axis]
                origin[axis] = lo[axis] + (hi[axis] - lo[axis]) * index / count
                part = chunk.slice_plane(origin, normal)
                origin[axis] = lo[axis] + (hi[axis] - lo[axis]) * (index + 1) / count
                sections.append(part.slice_plane(origin, -normal))
        chunks = sections
    spheres = []
    for index, chunk in enumerate(chunks):
        if len(chunk.vertices) == 0:
            raise ValueError(f"{name} section {index + 1} is empty")
        center, _ = trimesh.nsphere.minimum_nsphere(chunk.vertices)
        center = np.round(center, 6)
        # Recompute after rounding the center, then round the radius outward.
        radius = np.linalg.norm(chunk.vertices - center, axis=1).max() + 0.001
        radius = float(np.ceil(radius * 10000) / 10000)
        if frame_tag:
            spheres.append(dict(parent=name, frame=f"{name}_{frame_tag}_{index + 1}",
                                center=center.tolist(), radius=radius))
            continue
        frame = f"{name}_sphere{index + 1}"
        # Keep the existing palm prefix used by the self-collision pairs.
        if name == "left_link_base" and index >= 2:
            frame = f"palm_sphere{index - 1}" if index > 2 else "palm_sphere"
        spheres.append(dict(parent=name, frame=frame, center=center.tolist(), radius=radius))
    return spheres


def fit_v2(source: Path, extra_links: dict | None = None) -> dict:
    """v2 profile (see V2_SECTIONS). Distal links include their *_tip mesh; the section
    farthest from the joint (the fingertip cap) is shrunk by V2_TIP_ALLOWANCE."""
    root = ET.parse(source).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    spheres = []
    for name, ranked_bins in {**V2_SECTIONS, **(extra_links or {})}.items():
        mesh = distal_with_tip(root, links, source, name) if name.endswith("_4") else load_visual_mesh(links[name], source)
        lo, hi = mesh.bounds
        axes = np.argsort(hi - lo)[::-1]
        bins = [(int(axes[rank]), count) for rank, count in ranked_bins]
        fitted = section_spheres(name, mesh, bins, frame_tag="v2")
        if name.endswith("_4") and len(fitted) > 1:
            tip = max(fitted, key=lambda s: np.linalg.norm(s["center"]))
            tip["radius"] = float(np.ceil((tip["radius"] - 0.001 - V2_TIP_ALLOWANCE) * 10000) / 10000)
            tip["tip_allowance_m"] = V2_TIP_ALLOWANCE
        spheres.extend(fitted)
    if any(V2_PALM_OFFSET):
        for sp in spheres:
            if sp["parent"] == "left_link_base":
                sp["fitted_center"] = sp["center"]
                sp["center"] = (np.round(np.array(sp["center"]) + np.array(V2_PALM_OFFSET), 6)).tolist()
    if V2_PALM_SHRINK:
        palm = sorted((sp for sp in spheres if sp["parent"] == "left_link_base"), key=lambda sp: -sp["radius"])
        for sp in palm[:V2_PALM_SHRINK[0]]:
            sp["enclosing_radius"] = sp["radius"]
            sp["radius"] = float(np.ceil(sp["radius"] * V2_PALM_SHRINK[1] * 10000) / 10000)
    by_parent = {}
    for sp in spheres:
        by_parent.setdefault(sp["parent"], []).append(sp)
    by_frame = {sp["frame"]: sp for sp in spheres}

    def set_radius(sp, radius):
        sp.setdefault("enclosing_radius", sp["radius"])
        sp["radius"] = float(radius)
        sp.pop("tip_allowance_m", None)

    if V2_MIDDLE_MATCH_DISTAL:
        inner = [sp for sp in by_parent["left_link_5_4"] if "tip_allowance_m" not in sp]
        for sp in by_parent["left_link_5_3"]:
            set_radius(sp, inner[0]["radius"])
    if V2_FINGER_EQUAL:
        for f in range(2, 5):
            radius = by_frame[f"left_link_{f}_2_v2_2"]["radius"]
            for seg in (2, 3, 4):
                for sp in by_parent[f"left_link_{f}_{seg}"]:
                    set_radius(sp, radius)
    thumb = [sp for seg in (2, 3, 4) for sp in by_parent.get(f"left_link_1_{seg}", [])]
    if thumb:
        radius = V2_THUMB_RADIUS or float(np.median([sp.get("enclosing_radius", sp["radius"]) for sp in thumb]))
        for sp in thumb:
            set_radius(sp, np.ceil(radius * 10000) / 10000)
    overrides = yaml.safe_load(V2_OVERRIDES_FILE.read_text()) if V2_OVERRIDES_FILE.exists() else None
    for frame, change in (overrides or {}).items():
        if frame not in by_frame:
            raise KeyError(f"{V2_OVERRIDES_FILE.name}: unknown sphere {frame!r}")
        sp = by_frame[frame]
        if change.get("delete"):
            spheres.remove(sp)
            continue
        if change.get("radius") is not None:
            set_radius(sp, change["radius"])
        if change.get("offset") is not None:
            sp["center"] = np.round(np.array(sp["center"]) + np.array(change["offset"], float), 6).tolist()
        sp["override"] = change
    return dict(margin_m=0.001, method="clipped_mesh_minimum_enclosing_spheres", profile="v2",
                tip_allowance_m=V2_TIP_ALLOWANCE, arm_spheres=[], spheres=spheres)


def fit_s30(source: Path) -> dict:
    """30-sphere hand profile (see module docstring); arm spheres are not refitted."""
    root = ET.parse(source).getroot()
    links = {link.get("name"): link for link in root.findall("link")}
    spheres = []
    for name in HAND_LINKS:
        mesh = distal_with_tip(root, links, source, name) if name.endswith("_4") else load_visual_mesh(links[name], source)
        lo, hi = mesh.bounds
        axes = np.argsort(hi - lo)[::-1]
        if name == "left_link_base":
            bins = [(int(axes[0]), 2), (int(axes[1]), 2)]
        else:
            finger, seg = int(name.split("_")[2]), int(name.split("_")[3])
            bins = [(int(axes[0]), S30_SECTIONS[(finger, seg)])]
        spheres.extend(section_spheres(name, mesh, bins, frame_tag="s30"))
    return dict(margin_m=0.001, method="clipped_mesh_minimum_enclosing_spheres",
                profile="s30", arm_spheres=[], spheres=spheres)


def fit(source: Path) -> dict:
    root = ET.parse(source).getroot()
    links = {link.get("name"): link for link in root.findall("link")}

    def hand_mesh(name: str) -> trimesh.Trimesh:
        if name.endswith("_4"):
            return distal_with_tip(root, links, source, name)
        return load_visual_mesh(links[name], source)

    def fitted(names: tuple[str, ...], bins_for, mesh_for) -> list[dict]:
        spheres = []
        for name in names:
            mesh = mesh_for(name)
            lo, hi = mesh.bounds
            axes = np.argsort(hi - lo)[::-1]
            spheres.extend(section_spheres(name, mesh, bins_for(name, axes)))
        return spheres

    def hand_bins(name: str, axes) -> list[tuple[int, int]]:
        if name == "left_link_base":
            return [(axes[0], 3), (axes[1], 2)]
        if name.endswith("_4"):
            return [(int(axes[0]), DISTAL_SECTIONS)]
        return [(axes[0], 2)]

    hand = fitted(HAND_LINKS, hand_bins, hand_mesh)
    arm = fitted(
        ARM_LINKS,
        lambda _name, axes: [(int(axes[0]), ARM_SECTIONS)],
        lambda name: load_visual_mesh(links[name], source),
    )
    return dict(
        margin_m=0.001,
        method="clipped_mesh_minimum_enclosing_spheres",
        arm_sections=ARM_SECTIONS,
        arm_spheres=arm,
        spheres=hand,
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", type=Path, required=True)
    parser.add_argument("--profile", choices=("full", "s30", "v2"), default="full",
                        help="full = 41 hand spheres (default); s30 = 30 hand spheres; "
                             "v2 = knuckle-to-tip cover incl. left_link_<f>_1 (needs --src ..._w_mount.urdf)")
    parser.add_argument("--mount_spheres", type=int, default=0,
                        help="v2 only: spheres on left_link_mount (0 = rely on the wrist_3 / palm spheres)")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    if args.output is None:
        stem = "hand_collision_spheres" + ({"s30": "_s30", "v2": "_v2"}.get(args.profile, ""))
        args.output = Path(__file__).with_name(f"{stem}.json")
    if args.profile == "v2":
        extra = {"left_link_mount": [(0, args.mount_spheres)]} if args.mount_spheres else None
        result = fit_v2(args.src.resolve(), extra)
    else:
        result = (fit_s30 if args.profile == "s30" else fit)(args.src.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(f"{len(result['spheres'])} hand spheres, {len(result['arm_spheres'])} refitted arm spheres")
    names = list(dict.fromkeys(s["parent"] for s in result["spheres"])) if result.get("profile") == "v2" \
        else ((*ARM_LINKS, *HAND_LINKS) if result["arm_spheres"] else HAND_LINKS)
    for name in names:
        pool = result["arm_spheres"] if name in ARM_LINKS else result["spheres"]
        radii = [round(s["radius"] * 1000, 1) for s in pool if s["parent"] == name]
        print(f"{name}: {len(radii)} spheres, radii {radii} mm")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
