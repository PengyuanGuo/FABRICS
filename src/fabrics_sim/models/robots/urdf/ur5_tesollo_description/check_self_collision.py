#!/usr/bin/env python3
"""Find collider pairs that overlap when the URDF is imported into Isaac Sim with self-collision on.

Reproduces what the Isaac URDF importer does before PhysX sees the robot:
  * fixed joints are merged into the parent body (merge_fixed_joints=True),
  * every collision mesh becomes its CONVEX HULL (collider_type="convex_hull"),
  * PhysX filters only pairs of bodies that are directly connected by a joint.

For every remaining body pair it reports the penetration depth of the convex hulls (what
PhysX will push apart) and, for reference, whether the raw meshes also intersect.  Pairs that
penetrate at the zero pose are the ones that make the articulation explode.

    python3 check_self_collision.py                      # ur5_dg5fs_left.urdf at q = 0
    python3 check_self_collision.py --urdf ur5_dg5fs_right.urdf
    python3 check_self_collision.py --sweep              # also every per-joint sweep pose used by the Isaac test
    python3 check_self_collision.py --raw                # use raw meshes instead of convex hulls (convex_decomposition-like)

Needs: yourdfpy, trimesh, python-fcl  (pip install yourdfpy trimesh python-fcl)
"""
from __future__ import annotations

import argparse
import itertools
import json
from collections import defaultdict
from pathlib import Path

import fcl
import numpy as np
import trimesh
import yourdfpy

HERE = Path(__file__).resolve().parent


def merged_bodies(urdf: yourdfpy.URDF) -> tuple[dict[str, str], set[frozenset[str]]]:
    """link -> body(root of its fixed-joint cluster); and the set of body pairs joined by a movable joint."""
    parent_fixed = {}
    for j in urdf.robot.joints:
        if j.type == "fixed":
            parent_fixed[j.child] = j.parent

    def body_of(link: str) -> str:
        while link in parent_fixed:
            link = parent_fixed[link]
        return link

    body = {l.name: body_of(l.name) for l in urdf.robot.links}
    adjacent = {frozenset((body[j.parent], body[j.child])) for j in urdf.robot.joints if j.type != "fixed"}
    return body, adjacent


def collision_geoms(urdf: yourdfpy.URDF, use_hull: bool):
    """{link: [(trimesh, local_transform, label)]} from <collision> elements."""
    out = defaultdict(list)
    for link in urdf.robot.links:
        for k, c in enumerate(link.collisions):
            g = c.geometry
            if g.mesh is not None:
                m = trimesh.load(urdf._filename_handler(g.mesh.filename), force="mesh")
                if g.mesh.scale is not None:
                    m.apply_scale(g.mesh.scale)
                label = Path(g.mesh.filename).name
            elif g.box is not None:
                m, label = trimesh.creation.box(g.box.size), "box"
            elif g.cylinder is not None:
                m, label = trimesh.creation.cylinder(g.cylinder.radius, g.cylinder.length), "cylinder"
            elif g.sphere is not None:
                m, label = trimesh.creation.icosphere(radius=g.sphere.radius), "sphere"
            else:
                continue
            if use_hull:
                m = m.convex_hull
            T = c.origin if c.origin is not None else np.eye(4)
            out[link.name].append((m, T, f"{link.name}[{k}:{label}]"))
    return out


def fcl_obj(mesh: trimesh.Trimesh, T: np.ndarray) -> fcl.CollisionObject:
    bvh = fcl.BVHModel()
    bvh.beginModel(len(mesh.vertices), len(mesh.faces))
    bvh.addSubModel(mesh.vertices.astype(np.float64), mesh.faces.astype(np.int64))
    bvh.endModel()
    return fcl.CollisionObject(bvh, fcl.Transform(T[:3, :3], T[:3, 3]))


def check_pose(urdf, q, geoms, body, adjacent, contact_offset):
    """Return [(depth, bodyA, bodyB, geomA, geomB)] for non-adjacent body pairs closer than contact_offset."""
    urdf.update_cfg(q)
    objs = defaultdict(list)  # body -> [(fcl obj, label)]
    for link, items in geoms.items():
        T_link = urdf.get_transform(link)
        for m, T_local, label in items:
            objs[body[link]].append((fcl_obj(m, T_link @ T_local), label))
    hits = []
    for a, b in itertools.combinations(sorted(objs), 2):
        if frozenset((a, b)) in adjacent:
            continue
        for (oa, la), (ob, lb) in itertools.product(objs[a], objs[b]):
            creq = fcl.CollisionRequest(num_max_contacts=1, enable_contact=True)
            cres = fcl.CollisionResult()
            if fcl.collide(oa, ob, creq, cres) > 0:
                depth = max(c.penetration_depth for c in cres.contacts)
                hits.append((depth, a, b, la, lb))
                continue
            dres = fcl.DistanceResult()
            d = fcl.distance(oa, ob, fcl.DistanceRequest(), dres)
            if d < contact_offset:
                hits.append((-d, a, b, la, lb))
    return sorted(hits, reverse=True)


def sweep_poses(urdf, arm_amplitude=1.0, hand_fraction=0.9):
    """Zero pose + the per-joint lower/upper poses used by test_ur5_tesollo_urdf_joints.py."""
    names = urdf.actuated_joint_names
    J = len(names)
    poses = [("zero", np.zeros(J))]
    for i, n in enumerate(names):
        j = urdf.joint_map[n]
        lo, hi = j.limit.lower, j.limit.upper
        if n.startswith("ur5_"):
            lo_t, hi_t = max(lo, -arm_amplitude), min(hi, arm_amplitude)
        else:
            f = hand_fraction
            lo_t = lo * f if lo < 0 else lo + (1 - f) * (hi - lo)
            hi_t = hi * f if hi > 0 else hi - (1 - f) * (hi - lo)
        for label, v in (("lower", lo_t), ("upper", hi_t)):
            q = np.zeros(J); q[i] = v
            poses.append((f"{n}={v:+.3f} ({label})", q))
    return poses


def sine_poses(urdf, n=48, arm_amplitude=1.0, hand_fraction=0.9):
    """Poses along the all-joint sinusoid of test_ur5_tesollo_urdf_joints.py (full and half amplitude)."""
    sweep = sweep_poses(urdf, arm_amplitude, hand_fraction)[1:]
    J = len(urdf.actuated_joint_names)
    lo = np.array([q[i] for i, (_, q) in enumerate(sweep[0::2])])
    hi = np.array([q[i] for i, (_, q) in enumerate(sweep[1::2])])
    mid, amp = 0.5 * (lo + hi), 0.5 * (hi - lo)
    phase = np.linspace(0, 2 * np.pi, J)
    poses = []
    for ramp in (1.0, 0.5):
        for k in range(n):
            th = 2 * np.pi * k / n
            poses.append((f"sine ramp={ramp} theta={th:.2f}", mid + ramp * amp * np.sin(th + phase)))
    return poses


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", type=Path, default=HERE / "ur5_dg5fs_left.urdf")
    ap.add_argument("--raw", action="store_true", help="raw meshes instead of convex hulls")
    ap.add_argument("--sweep", action="store_true", help="also check every per-joint sweep pose")
    ap.add_argument("--sine", action="store_true", help="also check poses along the all-joint sine trajectory of the Isaac test")
    ap.add_argument("--hand_limit_fraction", type=float, default=0.9, help="hand sweep goes to +-fraction of the limits (1.0 = full range)")
    ap.add_argument("--contact_offset", type=float, default=0.002, help="report pairs closer than this [m] (PhysX contact offset)")
    ap.add_argument("--min_depth", type=float, default=0.0, help="only report penetrations deeper than this [m]")
    ap.add_argument("--json", type=Path, default=None, help="write {body: [bodies to filter]} adjacency map of all penetrating pairs")
    ap.add_argument("--json_margin", type=float, default=0.0,
                    help="also include pairs whose hulls come within this gap [m] (PhysX cooks coarser hulls than trimesh)")
    args = ap.parse_args()

    urdf = yourdfpy.URDF.load(args.urdf, load_meshes=False, load_collision_meshes=False, build_collision_scene_graph=False)
    body, adjacent = merged_bodies(urdf)
    geoms = collision_geoms(urdf, use_hull=not args.raw)
    clusters = defaultdict(list)
    for l, b in body.items():
        if l != b:
            clusters[b].append(l)
    print(f"[urdf] {args.urdf.name}: {len(urdf.robot.links)} links -> {len(set(body.values()))} bodies after fixed-joint merge")
    for b, ls in clusters.items():
        print(f"  {b} <- {ls}")
    print(f"[collider] {'raw meshes' if args.raw else 'convex hulls (Isaac importer default)'}; "
          f"{sum(len(v) for v in geoms.values())} collision shapes; adjacent (joint-connected) body pairs are skipped")

    poses = [("zero", np.zeros(len(urdf.actuated_joint_names)))]
    if args.sweep:
        poses += sweep_poses(urdf, hand_fraction=args.hand_limit_fraction)[1:]
    if args.sine:
        poses += sine_poses(urdf, hand_fraction=args.hand_limit_fraction)
    worst = {}  # pair -> (depth, pose, la, lb)
    for label, q in poses:
        for depth, a, b, la, lb in check_pose(urdf, q, geoms, body, adjacent, args.contact_offset):
            if depth < args.min_depth:
                continue
            key = (a, b)
            if key not in worst or depth > worst[key][0]:
                worst[key] = (depth, label, la, lb)
        if label == "zero":
            zero_hits = [(k, v) for k, v in worst.items()]
            print(f"\n[zero pose] {len(zero_hits)} overlapping / near-contact body pairs")
            for (a, b), (d, _, la, lb) in sorted(zero_hits, key=lambda kv: -kv[1][0]):
                kind = f"PENETRATION {d*1000:6.1f} mm" if d > 0 else f"gap {-d*1000:5.1f} mm"
                print(f"  {kind}  {a:<24} x {b:<24}   ({la} / {lb})")

    if args.sweep or args.sine:
        print(f"\n[sweep] worst overlap per body pair over {len(poses)} poses")
        for (a, b), (d, pose, la, lb) in sorted(worst.items(), key=lambda kv: -kv[1][0]):
            kind = f"PENETRATION {d*1000:6.1f} mm" if d > 0 else f"gap {-d*1000:5.1f} mm"
            print(f"  {kind}  {a:<24} x {b:<24}  at {pose}")

    if args.json:
        adjacency = defaultdict(list)
        for (a, b), (d, *_rest) in worst.items():
            if d > -args.json_margin:
                adjacency[a].append(b)
                adjacency[b].append(a)
        args.json.write_text(json.dumps({k: sorted(v) for k, v in sorted(adjacency.items())}, indent=2))
        print(f"\n[json] wrote {sum(len(v) for v in adjacency.values()) // 2} pairs (penetrating or within {args.json_margin*1000:.0f} mm) to {args.json}")


if __name__ == "__main__":
    main()
