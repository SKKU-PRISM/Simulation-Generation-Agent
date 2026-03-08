# OpenArm URDF

This directory contains the canonical repo-local OpenArm URDF used by the
data-collection Pinocchio IK path.

- Source repository: `https://github.com/enactic/openarm_description`
- Pinned ref: `79cd8b7cb5a22a0ce6de90e860d120cbc33085ac`
- Source xacro: `urdf/robot/v10.urdf.xacro`
- Render arguments: `arm_type:=v10 body_type:=v10 ee_type:=openarm_hand hand:=true ros2_control:=false bimanual:=false`

The rendered URDF is intended for kinematics and keeps the upstream
`package://openarm_description/...` mesh URIs. Regenerate it with:

```bash
python3 scripts/build_openarm_local_urdf.py
```
