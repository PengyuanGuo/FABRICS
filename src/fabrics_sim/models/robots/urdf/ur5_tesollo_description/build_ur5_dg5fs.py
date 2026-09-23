#!/usr/bin/env python3
"""ur5.urdf + Tesollo DG5F-S hand -> ur5_dg5fs_<side>.urdf

usage: python3 build_ur5_dg5fs.py left|right [--hand-src PATH]

- UR5 arm: 'world' 링크와 base_joint 제거 (base_link 가 root), 링크/조인트에 ur5_ 접두어
- Tesollo hand (mount 없는 버전): 링크/조인트에 <side>_ 접두어
- 결합: ur5_tool0 -> <side>_link_base (fixed). ATTACH_ORIGIN 참고
- mesh 경로: 이 폴더 기준 상대경로 (meshes/ur5/..., meshes/dg5fs_<side>/...)
"""
import argparse
import math
import re
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARM_SRC = HERE / "ur5.urdf"
DEFAULT_HAND_SRC = {
    "left": HERE / "dg5fs_left.urdf",
    "right": HERE / "dg5fs_right.urdf",
}
ARM_PREFIX = "ur5_"
ARM_DROP_LINKS, ARM_DROP_JOINTS = {"world"}, {"base_joint"}

# tool0 -> link_base.  손목을 -90deg(y축) 꺾은 사용자 값 (xyz 0.05 0 0.04, mount 두께 0.015 포함) 에서
# mount 링크를 제거하면서 mount->base 오프셋 0.015 를 합성한 값. link_base 의 최종 위치는 동일.
ATTACH_ORIGIN = {"xyz": "0.035 0 0.04", "rpy": "0 -1.57 0"}

# 손 collider 정책 (기본값 = 검증된 설정, 2026-09-02)
#  - 손바닥 link_base: mesh + SDF (오목한 형상이라 convex hull 이면 knuckle/엄지가 파고들어 self-collision 시 발산).
#    <collision> 안의 <sdf .../> 마커를 scene_utils._apply_urdf_sdf_collision_markers 가 Isaac import 후 SDF 로 바꿔준다.
#  - 손가락 마디/손끝: 원본 mesh, convex hull (마디는 거의 볼록해서 hull 과 mesh 가 같음). 물체 접촉면이므로 단순화하지 않음.
#  - knuckle link_x_1: <collision> 제거 (HAND_LINKS_WITHOUT_COLLISION). sharpa URDF 의 CMC_VL/MCP_VL 과 같은 취지.
#  - self-collision ON 에서 abduction 은 이웃 손가락에 막히는 것이 의도된 동작이며 필터하지 않는다.
#  - --hand-sdf all 은 Tesollo 제공 USD 와 동일(26개 전부 SDF). 접촉 거동은 같고 step 비용이 2~3배.
#  확인 도구: check_self_collision.py, isaacsimenvs/tests/test_ur5_tesollo_urdf_joints.py --self_collision
# 기본값: knuckle 5개 collider 제거 (손바닥 안에 묻혀 있어 물체와 닿지 않고, hull 로 만들면 이웃/손바닥과 항상 겹침)
HAND_LINKS_WITHOUT_COLLISION: set[str] = {"link_1_1", "link_2_1", "link_3_1", "link_4_1", "link_5_1"}
# 어느 손 링크에 SDF 마커를 붙일지. "all" = Tesollo USD 와 동일(26개 전부), "palm" = link_base 만
# (손가락은 convex hull; 영점 자세 파고듦은 손바닥 hull 때문이었으므로 비용 대비 절충안), "none" = 전부 hull.
HAND_SDF_MODE_LINKS = {"all": None, "palm": {"link_base"}, "none": set()}

# 손 조인트 한계 덮어쓰기 (접두어 없는 joint 이름 -> (lower, upper) [rad]), side 별. 기본은 비움.
# 예: 엄지 회전 joint_1_1 은 -57°/+17° 를 넘으면 엄지 첫마디가 손바닥 mesh 안으로 들어간다 (check_self_collision.py 스캔).
#     그런 자세는 실제 손도 못 만들고 SDF 손바닥 안에서 출발하는 접촉은 시뮬레이션을 발산시키므로, 여기서 잘라내면 된다.
HAND_JOINT_LIMIT_OVERRIDES: dict[str, dict[str, tuple[float, float]]] = {"left": {}, "right": {}}

# ---- "simple" collider mode (--hand-collider simple) ----
# 목적: 물체와 닿는 면(손끝 패드, 손바닥)만 원래 mesh 로 두고 나머지는 단순 형상으로 바꿔서,
# self-collision 을 켠 채로 joint range 끝까지 움직일 수 있게 한다 (sharpa asset 과 같은 취지).
#  - knuckle link_x_1: collider 제거 (손바닥 안에 묻혀 있고 물체와 닿지 않음)
#  - 마디 link_x_2, link_x_3, link_x_4, 엄지 link_1_2: 링크 원점에서 자식 joint 원점까지 잇는 원기둥(캡슐 역할).
#    반지름은 mesh 단면(r50 9.4 mm, r90 11.7 mm)보다 얇게 잡아 abduction 한계에서 옆 손가락과 안 닿게 한다.
#  - link_x_tip: mesh 유지 (convex hull; 손끝 패드는 볼록해서 hull 과 거의 같음)
#  - link_base: mesh 유지 + SDF (손바닥 접촉면)
SIMPLE_KNUCKLE_LINKS = {"link_1_1", "link_2_1", "link_3_1", "link_4_1", "link_5_1"}
SIMPLE_SEGMENT_RADIUS = {2: 0.0090, 3: 0.0090, 4: 0.0080}   # link_x_<segment> -> radius [m]
SIMPLE_SEGMENT_END_GAP = 0.002                                # 원기둥 길이 = 관절간 거리 - 2*gap
SIMPLE_SDF_LINKS = {"link_base"}
# Tesollo dg5fs_15dof_*_physics.usd 의 PhysxSDFMeshCollisionAPI 값 그대로
HAND_SDF_PARAMS = {"resolution": "256", "margin": "0.01", "narrow_band_thickness": "0.01", "subgrid_resolution": "6"}


def rename(root: ET.Element, prefix: str) -> None:
    for el in root.iter():
        if el.tag in ("link", "joint") and el.get("name"):
            el.set("name", prefix + el.get("name"))
        if el.tag in ("parent", "child") and el.get("link"):
            el.set("link", prefix + el.get("link"))
        if el.tag == "mimic" and el.get("joint"):
            el.set("joint", prefix + el.get("joint"))


def _rpy_z_to(direction) -> tuple[float, float, float]:
    """roll/pitch/yaw that rotates the local z axis onto `direction` (URDF cylinder axis is z)."""
    d = [x / math.sqrt(sum(v * v for v in direction)) for x in direction]
    # rotate about y by pitch, then about x by roll: z -> (sin p cos r, -sin r, cos p cos r)
    roll = math.atan2(-d[1], math.sqrt(d[0] ** 2 + d[2] ** 2))
    pitch = math.atan2(d[0], d[2])
    return roll, pitch, 0.0


def simplify_hand_colliders(hand: ET.Element) -> list[str]:
    """Apply the 'simple' collider policy in place (names WITHOUT side prefix). Returns a log."""
    log = []
    child_origin = {}
    for j in hand.iter("joint"):
        o = j.find("origin")
        child_origin[j.find("parent").get("link")] = [float(v) for v in (o.get("xyz") if o is not None else "0 0 0").split()]
    for link in hand.iter("link"):
        name = link.get("name")
        m = re.fullmatch(r"link_(\d)_(\d|tip|base)", name)
        if m is None:
            continue
        seg = m.group(2)
        if name in SIMPLE_KNUCKLE_LINKS:
            for col in link.findall("collision"):
                link.remove(col)
            log.append(f"{name}: collider removed")
        elif seg in ("2", "3", "4"):
            p = child_origin[name]
            dist = math.sqrt(sum(x * x for x in p))
            r = SIMPLE_SEGMENT_RADIUS[int(seg)]
            length = max(dist - 2 * SIMPLE_SEGMENT_END_GAP, 2 * r)
            for col in link.findall("collision"):
                link.remove(col)
            col = ET.SubElement(link, "collision")
            ET.SubElement(col, "origin", {"xyz": " ".join(f"{x/2:.5f}" for x in p),
                                          "rpy": " ".join(f"{a:.6f}" for a in _rpy_z_to(p))})
            geo = ET.SubElement(col, "geometry")
            ET.SubElement(geo, "cylinder", {"radius": f"{r:.4f}", "length": f"{length:.4f}"})
            log.append(f"{name}: cylinder r={r*1000:.1f}mm L={length*1000:.1f}mm along child joint")
        elif seg == "base":
            for col in link.findall("collision"):
                if col.find("sdf") is None:
                    col.insert(0, ET.Element("sdf", dict(HAND_SDF_PARAMS)))
            log.append(f"{name}: mesh kept, SDF")
        else:  # tip
            log.append(f"{name}: mesh kept (convex hull)")
    return log


def rewrite_meshes(root: ET.Element, pattern: str, repl: str) -> None:
    for m in root.iter("mesh"):
        m.set("filename", re.sub(pattern, repl, m.get("filename")))


def build(side: str, hand_src: Path, hand_sdf: str = "all", out_name: str | None = None,
          hand_collider: str = "tesollo") -> Path:
    hand_prefix = f"{side}_"
    out = HERE / (out_name or f"ur5_dg5fs_{side}.urdf")
    sdf_links = HAND_SDF_MODE_LINKS[hand_sdf]

    arm = ET.parse(ARM_SRC).getroot()
    for el in list(arm):
        if (el.tag == "link" and el.get("name") in ARM_DROP_LINKS) or (
            el.tag == "joint" and el.get("name") in ARM_DROP_JOINTS
        ):
            arm.remove(el)
    rewrite_meshes(arm, r"^(?:.*/)?meshes/(?:ur5/)?(visual|collision)/([^/]+)$", r"meshes/ur5/\1/\2")
    rename(arm, ARM_PREFIX)

    hand = ET.parse(hand_src).getroot()
    rewrite_meshes(hand, r"^package://meshes/", "meshes/")
    for jname, (lo, hi) in HAND_JOINT_LIMIT_OVERRIDES[side].items():
        lim = next(j.find("limit") for j in hand.iter("joint") if j.get("name") == jname)
        print(f"  [limit] {jname}: ({lim.get('lower')}, {lim.get('upper')}) -> ({lo}, {hi})")
        lim.set("lower", f"{lo:.6f}"); lim.set("upper", f"{hi:.6f}")
    removed = []
    for link in hand.iter("link"):
        if link.get("name") in HAND_LINKS_WITHOUT_COLLISION:
            for col in link.findall("collision"):
                link.remove(col)
                removed.append(link.get("name"))
    assert set(removed) == HAND_LINKS_WITHOUT_COLLISION, f"collision removal mismatch: {removed}"
    if hand_collider == "simple":
        for line in simplify_hand_colliders(hand):
            print("  [simple]", line)
    else:
        for link in hand.iter("link"):
            if sdf_links is not None and link.get("name") not in sdf_links:
                continue
            for col in link.findall("collision"):
                if col.find("sdf") is None:
                    col.insert(0, ET.Element("sdf", dict(HAND_SDF_PARAMS)))
    rename(hand, hand_prefix)
    hand_links = {l.get("name") for l in hand.iter("link")}
    assert f"{hand_prefix}link_base" in hand_links, "hand source must contain link_base"
    assert f"{hand_prefix}link_mount" not in hand_links, "use the hand source WITHOUT mount"

    robot = ET.Element("robot", {"name": f"ur5_dg5fs_{side}"})
    robot.append(ET.Comment(" ===== UR5 arm (from ur5.urdf, Universal_Robots_ROS2_Description ur_type:=ur5) ===== "))
    robot.extend(list(arm))
    robot.append(ET.Comment(" ===== arm -> hand attachment (wrist bent -90deg about y; mount plate removed) ===== "))
    attach = ET.SubElement(robot, "joint", {"name": f"{ARM_PREFIX}tool0_to_{hand_prefix}link_base", "type": "fixed"})
    ET.SubElement(attach, "origin", ATTACH_ORIGIN)
    ET.SubElement(attach, "parent", {"link": f"{ARM_PREFIX}tool0"})
    ET.SubElement(attach, "child", {"link": f"{hand_prefix}link_base"})
    robot.append(ET.Comment(f" ===== Tesollo DG5F-S {side} hand (from {hand_src.name}); "
                            f"hand collider={hand_collider} SDF mode={hand_sdf} <sdf {HAND_SDF_PARAMS}/>; "
                            f"<collision> removed on {sorted(HAND_LINKS_WITHOUT_COLLISION)} ===== "))
    robot.extend(list(hand))

    ET.indent(robot, space="  ")
    out.write_text('<?xml version="1.0"?>\n' + ET.tostring(robot, encoding="unicode") + "\n")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("side", choices=["left", "right"])
    ap.add_argument("--hand-src", type=Path, default=None)
    ap.add_argument("--hand-sdf", choices=sorted(HAND_SDF_MODE_LINKS), default="palm",
                    help="palm (default): link_base only is SDF, finger links convex hull; all: Tesollo USD style; none: all hull")
    ap.add_argument("--out", default=None, help="output file name (default ur5_dg5fs_<side>.urdf)")
    ap.add_argument("--hand-collider", choices=["tesollo", "simple"], default="tesollo",
                    help="tesollo: original meshes (+SDF per --hand-sdf); simple: see SIMPLE_* above")
    ap.add_argument("--seg-radius", type=float, default=None,
                    help="simple mode: cylinder radius [m] for segments 2/3 (segment 4 uses radius - 1 mm)")
    ap.add_argument("--no-knuckle-colliders", action="store_true",
                    help="drop <collision> on link_x_1 (buried in the palm, never touches objects); works with any --hand-collider/--hand-sdf")
    a = ap.parse_args()
    if a.no_knuckle_colliders:
        HAND_LINKS_WITHOUT_COLLISION.update(SIMPLE_KNUCKLE_LINKS)
    if a.seg_radius is not None:
        SIMPLE_SEGMENT_RADIUS.update({2: a.seg_radius, 3: a.seg_radius, 4: a.seg_radius - 0.001})
    print("wrote", build(a.side, a.hand_src or DEFAULT_HAND_SRC[a.side], a.hand_sdf, a.out, a.hand_collider))
