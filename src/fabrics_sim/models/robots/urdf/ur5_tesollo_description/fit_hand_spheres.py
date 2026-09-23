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
"""

import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial.transform import Rotation
import trimesh


HAND_LINKS = ("left_link_base",) + tuple(
    f"left_link_{finger}_{segment}" for finger in range(1, 6) for segment in (2, 3, 4)
)
ARM_LINKS = ("ur5_upper_arm_link", "ur5_forearm_link")
ARM_SECTIONS = 4
DISTAL_SECTIONS = 3


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


def section_spheres(name: str, mesh: trimesh.Trimesh, bins: list[tuple[int, int]]) -> list[dict]:
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
        frame = f"{name}_sphere{index + 1}"
        # Keep the existing palm prefix used by the self-collision pairs.
        if name == "left_link_base" and index >= 2:
            frame = f"palm_sphere{index - 1}" if index > 2 else "palm_sphere"
        spheres.append(dict(parent=name, frame=frame, center=center.tolist(), radius=radius))
    return spheres


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
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("hand_collision_spheres.json"))
    args = parser.parse_args()
    result = fit(args.src.resolve())
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    for name in (*ARM_LINKS, *HAND_LINKS):
        pool = result["arm_spheres"] if name in ARM_LINKS else result["spheres"]
        radii = [round(s["radius"] * 1000, 1) for s in pool if s["parent"] == name]
        print(f"{name}: {len(radii)} spheres, radii {radii} mm")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
