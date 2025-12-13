#!/usr/bin/env python3
"""
Inspect a vehicle USD structure WITHOUT printing numeric parameter values.

Prints:
- articulation roots, rigid bodies, colliders
- for each joint: body0/body1, excludeFromArticulation, jointEnabled, axis token,
  presence of joint frame attrs (localPos/localRot), presence of limits (attr + LimitAPI),
  DriveAPI instance names (no values)
- suggests a feasible articulation spanning tree by treating joints as UNDIRECTED edges
  (PhysX-like), and lists loop edges / excluded edges (constraints).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from pxr import Usd, UsdGeom, UsdPhysics  # type: ignore


USD_PATH = "newton/examples/assets/rsl_vehicle/vehicle_rsl_improve_steering_suspension_wheel.usda"
ROOT_BODIES_SCOPE = "/Vehicle/Bodies"
ROOT_JOINTS_SCOPE = "/Vehicle/Joints"


def _get_single_target(rel: Usd.Relationship) -> Optional[str]:
    if not rel:
        return None
    targets = rel.GetTargets()
    if not targets:
        return None
    return str(targets[0])


def _has_attr(prim: Usd.Prim, attr_name: str) -> bool:
    a = prim.GetAttribute(attr_name)
    return bool(a) and a.HasAuthoredValueOpinion()


def _schema_name(prim: Usd.Prim) -> str:
    try:
        return prim.GetTypeName() or prim.GetName()
    except Exception:
        return prim.GetName()


def _is_joint_prim(prim: Usd.Prim) -> bool:
    try:
        return prim.IsA(UsdPhysics.Joint)
    except Exception:
        return False


def _bool_attr(prim: Usd.Prim, name: str, default: Optional[bool] = None) -> Optional[bool]:
    a = prim.GetAttribute(name)
    if not a or not a.HasAuthoredValueOpinion():
        return default
    try:
        v = a.Get()
        return bool(v)
    except Exception:
        return default


def _list_drive_instances(joint_prim: Usd.Prim) -> List[str]:
    # Report instance tokens only (no values).
    instances: List[str] = []
    for tok in ("linear", "angular", "transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
        try:
            api = UsdPhysics.DriveAPI.Get(joint_prim, tok)
            if api and api.GetPrim() and api.GetPrim().IsValid():
                # consider present if any common drive attrs are authored OR schema applied
                instances.append(tok)
        except Exception:
            pass
    # de-dupe, stable
    seen = set()
    out = []
    for x in instances:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def _list_limit_instances(joint_prim: Usd.Prim) -> List[str]:
    instances: List[str] = []
    for tok in ("transX", "transY", "transZ", "rotX", "rotY", "rotZ"):
        try:
            api = UsdPhysics.LimitAPI.Get(joint_prim, tok)
            if api and api.GetPrim() and api.GetPrim().IsValid():
                instances.append(tok)
        except Exception:
            pass
    return instances


@dataclass(frozen=True)
class JointInfo:
    path: str
    type_name: str
    body0: Optional[str]
    body1: Optional[str]
    enabled: Optional[bool]
    excluded: Optional[bool]
    axis_token: Optional[str]
    has_localPos0: bool
    has_localPos1: bool
    has_localRot0: bool
    has_localRot1: bool
    has_lower: bool
    has_upper: bool
    drive_instances: Tuple[str, ...]
    limit_instances: Tuple[str, ...]


def collect_joint_infos(stage: Usd.Stage) -> List[JointInfo]:
    out: List[JointInfo] = []
    for prim in stage.Traverse():
        if not prim.IsValid() or not _is_joint_prim(prim):
            continue
        if not str(prim.GetPath()).startswith(ROOT_JOINTS_SCOPE):
            continue

        body0 = _get_single_target(prim.GetRelationship("physics:body0"))
        body1 = _get_single_target(prim.GetRelationship("physics:body1"))

        enabled = _bool_attr(prim, "physics:jointEnabled", default=None)
        excluded = _bool_attr(prim, "physics:excludeFromArticulation", default=None)

        axis_token = None
        if prim.HasAttribute("physics:axis"):
            try:
                v = prim.GetAttribute("physics:axis").Get()
                axis_token = str(v) if v is not None else None
            except Exception:
                axis_token = None

        has_localPos0 = _has_attr(prim, "physics:localPos0")
        has_localPos1 = _has_attr(prim, "physics:localPos1")
        has_localRot0 = _has_attr(prim, "physics:localRot0")
        has_localRot1 = _has_attr(prim, "physics:localRot1")

        has_lower = _has_attr(prim, "physics:lowerLimit")
        has_upper = _has_attr(prim, "physics:upperLimit")

        drive_instances = tuple(_list_drive_instances(prim))
        limit_instances = tuple(_list_limit_instances(prim))

        out.append(
            JointInfo(
                path=str(prim.GetPath()),
                type_name=_schema_name(prim),
                body0=body0,
                body1=body1,
                enabled=enabled,
                excluded=excluded,
                axis_token=axis_token,
                has_localPos0=has_localPos0,
                has_localPos1=has_localPos1,
                has_localRot0=has_localRot0,
                has_localRot1=has_localRot1,
                has_lower=has_lower,
                has_upper=has_upper,
                drive_instances=drive_instances,
                limit_instances=limit_instances,
            )
        )
    out.sort(key=lambda j: j.path)
    return out


def find_articulation_roots(stage: Usd.Stage) -> List[str]:
    roots: List[str] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        try:
            api = UsdPhysics.ArticulationRootAPI(prim)
            if api and api.GetPrim() and api.GetPrim().IsValid():
                roots.append(str(prim.GetPath()))
        except Exception:
            pass
    roots.sort()
    return roots


def collect_rigid_bodies_and_colliders(stage: Usd.Stage) -> Tuple[List[str], List[str]]:
    bodies: List[str] = []
    colliders: List[str] = []
    for prim in stage.Traverse():
        if not prim.IsValid():
            continue
        try:
            if UsdPhysics.RigidBodyAPI(prim):
                bodies.append(str(prim.GetPath()))
        except Exception:
            pass
        try:
            if UsdPhysics.CollisionAPI(prim):
                colliders.append(str(prim.GetPath()))
        except Exception:
            pass
    bodies.sort()
    colliders.sort()
    return bodies, colliders


def build_spanning_tree(
    joints: List[JointInfo],
    root_body_path: str,
) -> Tuple[List[JointInfo], List[JointInfo], List[JointInfo]]:
    """
    Treat joints as UNDIRECTED edges and build a spanning tree from root_body_path.

    Returns:
        tree_edges: joints used in the tree
        loop_edges: enabled, non-excluded joints that would close cycles or remain unused
        excluded_edges: enabled joints explicitly excludedFromArticulation=True
    """
    def is_enabled(j: JointInfo) -> bool:
        return (j.enabled is None) or (j.enabled is True)

    excluded_edges = [j for j in joints if is_enabled(j) and (j.excluded is True)]
    candidate_edges = [j for j in joints if is_enabled(j) and (j.excluded is not True)]

    adj: Dict[str, List[JointInfo]] = {}
    for j in candidate_edges:
        if not j.body0 or not j.body1:
            continue
        adj.setdefault(j.body0, []).append(j)
        adj.setdefault(j.body1, []).append(j)

    visited: Set[str] = {root_body_path}
    used: Set[str] = set()
    tree_edges: List[JointInfo] = []
    loop_edges: List[JointInfo] = []

    queue: List[str] = [root_body_path]
    while queue:
        b = queue.pop(0)
        for j in adj.get(b, []):
            if j.path in used:
                continue
            if not j.body0 or not j.body1:
                continue
            other = j.body1 if b == j.body0 else j.body0
            used.add(j.path)
            if other in visited:
                loop_edges.append(j)
            else:
                tree_edges.append(j)
                visited.add(other)
                queue.append(other)

    # Any leftover candidate edges not used are either disconnected or additional constraints
    for j in candidate_edges:
        if j.path not in used:
            loop_edges.append(j)

    return tree_edges, loop_edges, excluded_edges


def main() -> None:
    print("=" * 88)
    print("USD open")
    print("=" * 88)

    usd_path = Path(USD_PATH)
    stage = Usd.Stage.Open(str(usd_path))
    if stage is None:
        raise RuntimeError(f"Failed to open USD: {USD_PATH}")

    print(f"Opened: {USD_PATH}")
    root_layer = stage.GetRootLayer()
    print(f"Root layer: {root_layer.realPath}")

    print("\n" + "=" * 88)
    print("Stage settings (tokens only)")
    print("=" * 88)
    try:
        up = UsdGeom.GetStageUpAxis(stage)
        print(f"Up axis: {up}")
    except Exception:
        print("Up axis: (unknown)")

    # Meters-per-unit: print presence only, no numeric values
    try:
        authored = stage.HasAuthoredMetadata("metersPerUnit")
    except Exception:
        authored = False
    print(f"metersPerUnit metadata authored: {authored}")

    print("\n" + "=" * 88)
    print("Articulation roots")
    print("=" * 88)
    roots = find_articulation_roots(stage)
    if roots:
        for r in roots:
            print(f"- {r}")
    else:
        print("(none)")

    print("\n" + "=" * 88)
    print("Rigid bodies and collision shapes (paths only)")
    print("=" * 88)
    bodies, colliders = collect_rigid_bodies_and_colliders(stage)
    print(f"Rigid bodies: {len(bodies)}")
    print(f"Colliders  : {len(colliders)}")

    wheel_like = [p for p in colliders if "/Wheel_" in p or "/Wheel" in p]
    if wheel_like:
        print("\nWheel-like collider prims:")
        for p in wheel_like:
            print(f"- {p}")

    print("\n" + "=" * 88)
    print("UsdPhysics joints (flags + presence only; no numeric values)")
    print("=" * 88)
    joints = collect_joint_infos(stage)
    for j in joints:
        print(f"{j.type_name:18s} {j.path}")
        print(f"  body0: {j.body0}")
        print(f"  body1: {j.body1}")
        print(f"  flags: excludeFromArticulation={j.excluded}  jointEnabled={j.enabled}")
        if j.axis_token is not None:
            print(f"  axis : {j.axis_token}")
        print(
            "  frames: "
            f"localPos0={j.has_localPos0} localRot0={j.has_localRot0} "
            f"localPos1={j.has_localPos1} localRot1={j.has_localRot1}"
        )
        print(f"  limits: lowerAttr={j.has_lower} upperAttr={j.has_upper} limitAPI={list(j.limit_instances)}")
        if j.drive_instances:
            print(f"  DriveAPI instances: {list(j.drive_instances)}")
        print()

    print("\n" + "=" * 88)
    print("Spanning-tree suggestion (UNDIRECTED joints; PhysX-like)")
    print("=" * 88)

    # choose root body: prefer an articulation root under /Vehicle/Bodies
    root_body = None
    for r in roots:
        if r.startswith(ROOT_BODIES_SCOPE):
            root_body = r
            break
    if root_body is None:
        root_body = f"{ROOT_BODIES_SCOPE}/Base"

    tree_edges, loop_edges, excluded_edges = build_spanning_tree(joints, root_body)

    print(f"Root body used for tree build: {root_body}\n")
    print(f"Tree edges chosen: {len(tree_edges)}")
    for j in tree_edges:
        print(f"- {j.path}  ({j.body0} <-> {j.body1})")

    print(f"\nExplicit EXCLUDED edges (constraints): {len(excluded_edges)}")
    for j in excluded_edges:
        print(f"- {j.path}  ({j.body0} <-> {j.body1})")

    print(f"\nLoop / leftover edges (also constraints if you want a pure tree): {len(loop_edges)}")
    for j in loop_edges:
        print(f"- {j.path}  ({j.body0} <-> {j.body1})")

    print("\n" + "=" * 88)
    print("Done")
    print("=" * 88)


if __name__ == "__main__":
    main()
