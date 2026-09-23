#!/usr/bin/env bash
# Universal_Robots_ROS2_Description 의 xacro 를 ROS 없이 flatten 해서 ur5.urdf 를 다시 만든다.
# 사용법: ./regenerate.sh [ur_type]   (기본 ur5. 예: ur5e, ur10 ...)
set -euo pipefail
UR_TYPE="${1:-ur5}"
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="${UR_REPO:-$HERE/../Universal_Robots_ROS2_Description}"
WORK="$(mktemp -d)"

# 1) xacro 만 설치된 임시 venv
python3 -m venv "$WORK/venv" && "$WORK/venv/bin/pip" -q install xacro pyyaml

# 2) 소스 복사 후 $(find ur_description) 을 절대경로로 치환 (ROS 패키지 검색 대체)
cp -r "$REPO/urdf" "$REPO/config" "$WORK/"
/usr/bin/grep -rlF '$(find ur_description)' "$WORK/urdf" | xargs sed -i "s|\$(find ur_description)|$WORK|g"

# 3) xacro 실행: ur_type 에 따라 config/<ur_type>/*.yaml 이 선택됨
"$WORK/venv/bin/xacro" "$WORK/urdf/ur.urdf.xacro" ur_type:="$UR_TYPE" name:="$UR_TYPE" -o "$WORK/raw.urdf"

# 4) mesh 경로를 package:// 에서 이 폴더 기준 상대경로로 변경, mesh/config 복사
mkdir -p "$HERE/meshes" "$HERE/config"
cp -r "$REPO/meshes/$UR_TYPE/visual" "$REPO/meshes/$UR_TYPE/collision" "$HERE/meshes/"
cp "$REPO/config/$UR_TYPE/"*.yaml "$HERE/config/"
sed "s|package://ur_description/meshes/$UR_TYPE/|meshes/|g" "$WORK/raw.urdf" > "$HERE/$UR_TYPE.urdf"

rm -rf "$WORK"
echo "generated: $HERE/$UR_TYPE.urdf"
