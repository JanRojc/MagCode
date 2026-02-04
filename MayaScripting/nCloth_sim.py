# Maya ground-truth cloth sim for AMASS (CMU) + TailorNet garment OBJs
# - Reads AMASS .npz via an external Python (with smplx+torch installed) to build a body-cache .npz (verts+faces+fps)
# - In Maya: loads garment OBJ, builds a body mesh, drives body vertices per-frame from cache, runs nCloth sim
# - Exports per-frame PLYs:
#     <OUT_ROOT>/CMU/<seq_num>/<seq_idx>/<gender>/<garment>/
#         body/body_0000.ply ...
#         cloth/pred_gar_0000.ply ...
# - Optional: playblast + ffmpeg mp4 output_0.mp4, output_1.mp4, ...

from maya.api import OpenMaya as om
import maya.cmds as cmds
import maya.mel as mel

import os
import os.path as osp
import math
import shutil
import stat
import tempfile
import subprocess
import struct

# ======================================================================================
# CONFIG
# ======================================================================================

# Root that contains CMU folder:
AMASS_ROOT = r"D:\ClothSim\AMASS"

# Folder with SMPL models for smplx.create(model_path=MODEL_FOLDER, model_type="smpl", ...)
MODEL_FOLDER = r"D:\ClothSim"  # must contain SMPL model files in the layout smplx expects

# External python (NOT Maya python) that has: numpy, torch, smplx installed
EXTERNAL_PYTHON = r"C:\Users\janr\Documents\MagCode\.venv_py310\Scripts\python.exe"

# Folder with TailorNet garment OBJs named:
#   tailornet_{garment}_{gender}_{sequence_num}.obj
GARMENT_OBJ_DIR = r"D:\ClothSim\ccraft_data\aux_data\garment_meshes\smpl"

# Where to write outputs
OUT_ROOT = r"D:\ClothSim\Results\Maya"

# Wrap control
# False: wrap only at frame 0 to dress, then bake + delete wrap history => "nCloth-only" motion (plus constraints)
# True: keep wrap live for all frames => "wrap + nCloth" (useful for comparison)
USE_WRAP_THROUGHOUT = False

# Sampling
TARGET_FPS = 30
DEFAULT_MOCAP_FPS = 30  # if mocap_framerate missing in npz

# Simulation
CLEAR_SCENE_EACH_RUN = True
GRAVITY = 9.8
NUCLEUS_SUBSTEPS = 8
NUCLEUS_MAX_COLLISION_ITERS = 8

# Cloth settings (reasonable defaults; tweak)
CLOTH_STRETCH_RESIST = 50.0
CLOTH_COMPRESSION_RESIST = 50.0
CLOTH_BEND_RESIST = 0.1
CLOTH_DAMP = 0.1
CLOTH_THICKNESS = 0.002  # meters-ish scale

# Collision
COLLIDE_THICKNESS = 0.003
SELF_COLLIDE = True

# Pinning (top Y% in initial garment pose)
AUTO_PIN_TOP_RATIO = 0.02

# Export
EXPORT_PAD = 4

# Optional MP4 rendering
MAKE_MP4 = False
PLAYBLAST_W = 1024
PLAYBLAST_H = 1024
FFMPEG_BIN = "ffmpeg"  # must be in PATH for Maya process


# ======================================================================================
# External exporter script (runs in EXTERNAL_PYTHON env)
# Produces a compressed npz with: verts (T,V,3), faces (F,3), fps, skip, mocap_fps, gender
# ======================================================================================

_EXPORTER_CODE = r"""
import os, os.path as osp
import numpy as np

if not hasattr(np, "bool"):
    np.bool = np.bool_
if not hasattr(np, "int"):
    np.int = int
if not hasattr(np, "float"):
    np.float = float
if not hasattr(np, "complex"):
    np.complex = complex
if not hasattr(np, "object"):
    np.object = object
if not hasattr(np, "unicode"):
    np.unicode = str
if not hasattr(np, "str"):
    np.str = str

import torch
import smplx

DEFAULT_MOCAP_FPS = int(os.environ.get("AMASS_DEFAULT_MOCAP_FPS", "30"))

def load_amass_raw(poses_path: str):
    if not osp.isfile(poses_path):
        raise FileNotFoundError(poses_path)
    return dict(np.load(poses_path, allow_pickle=True))

def load_shape(seq_dir: str, raw: dict, force_gender=None, default_gender="female"):
    betas = raw.get("betas", None)
    gender = raw.get("gender", None)

    shp = osp.join(seq_dir, "shape.npz")
    if osp.exists(shp):
        s = dict(np.load(shp, allow_pickle=True))
        if "betas" in s:
            betas = s["betas"]
        if "gender" in s and gender is None:
            gender = s["gender"]

    if betas is None:
        betas = np.zeros(16, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)

    if force_gender is not None:
        gender = force_gender
    if gender is None:
        gender = default_gender
    else:
        gender = str(gender)

    return betas, gender

def parse_smpl_params(raw: dict):
    poses = np.asarray(raw["poses"], dtype=np.float32)
    if poses.ndim != 2 or poses.shape[1] < 72:
        raise ValueError(f"Unexpected poses shape {poses.shape}, expected (T,>=72)")

    global_orient = poses[:, :3]
    body_pose = poses[:, 3:72]  # 69

    trans = raw.get("trans", None)
    if trans is None:
        trans = np.zeros((poses.shape[0], 3), dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)

    fps = raw.get("mocap_framerate", None)
    if fps is None:
        fps = raw.get("mocap_fps", None)
    if fps is None:
        fps = DEFAULT_MOCAP_FPS
    fps = int(np.array(fps).item())

    return global_orient, body_pose, trans, fps

def export_body_cache(poses_path, model_folder, out_npz, target_fps=30, force_gender=None, default_gender="female"):
    raw = load_amass_raw(poses_path)
    seq_dir = osp.dirname(poses_path)

    betas, gender = load_shape(seq_dir, raw, force_gender=force_gender, default_gender=default_gender)
    go, bp, tr, mocap_fps = parse_smpl_params(raw)

    skip = int(round(float(mocap_fps) / float(target_fps)))
    skip = max(skip, 1)

    go = go[::skip]
    bp = bp[::skip]
    tr = tr[::skip]
    T = go.shape[0]

    model = smplx.create(
        model_path=model_folder,
        model_type="smpl",
        gender=gender,
        use_pca=False,
        batch_size=T,
    )

    betas_t = np.tile(betas[None, :], (T, 1)).astype(np.float32)
    betas_t = betas_t[:, :model.num_betas]

    with torch.no_grad():
        out = model(
            betas=torch.from_numpy(betas_t).float(),
            body_pose=torch.from_numpy(bp).float(),
            global_orient=torch.from_numpy(go).float(),
            transl=torch.from_numpy(tr).float(),
            return_verts=True,
        )

    verts = out.vertices.detach().cpu().numpy().astype(np.float32)  # (T,V,3)
    faces = model.faces.astype(np.int32)

    os.makedirs(osp.dirname(out_npz), exist_ok=True)
    np.savez_compressed(
        out_npz,
        verts=verts,
        faces=faces,
        fps=int(target_fps),
        skip=int(skip),
        mocap_fps=int(mocap_fps),
        gender=str(gender),
        poses_path=str(poses_path),
        global_orient=go.astype(np.float32),
        transl=tr.astype(np.float32),
    )
    print("Wrote:", out_npz, "verts:", verts.shape, "faces:", faces.shape, "fps:", target_fps, "skip:", skip)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--poses_path", required=True)
    ap.add_argument("--model_folder", required=True)
    ap.add_argument("--out_npz", required=True)
    ap.add_argument("--target_fps", type=int, default=30)
    ap.add_argument("--force_gender", default=None)
    ap.add_argument("--default_gender", default="female")
    args = ap.parse_args()
    export_body_cache(args.poses_path, args.model_folder, args.out_npz, args.target_fps, args.force_gender, args.default_gender)
"""


# ======================================================================================
# Helpers: filesystem
# ======================================================================================

def _ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)

def _on_rmtree_error(func, path, exc_info):
    # Handle Windows permission bits
    try:
        os.chmod(path, stat.S_IWRITE)
        func(path)
    except Exception:
        pass

def _rm_tree_if_exists(p: str):
    if osp.isdir(p):
        shutil.rmtree(p, onerror=_on_rmtree_error)

def _next_available_mp4(out_dir: str, base="output", ext=".mp4") -> str:
    i = 0
    while True:
        cand = osp.join(out_dir, f"{base}_{i}{ext}")
        if not osp.exists(cand):
            return cand
        i += 1

# ======================================================================================
# Helpers: AMASS + garment path resolution
# ======================================================================================

def find_amass_poses_file(amass_root: str, sequence_num: str, sequence_idx: str) -> str:
    p = osp.join(amass_root, "CMU", str(sequence_num), f"{sequence_num}_{sequence_idx}_poses.npz")
    if not osp.isfile(p):
        raise FileNotFoundError(f"AMASS poses not found: {p}")
    return p

def find_garment_obj(garment_obj_dir: str, garment: str, gender: str, sequence_num: str) -> str:
    fn = f"tailornet_{garment}_{gender}_{sequence_num}.obj"
    p = osp.join(garment_obj_dir, fn)
    if not osp.isfile(p):
        raise FileNotFoundError(f"Garment OBJ not found: {p}")
    return p

# ======================================================================================
# Helpers: body cache creation
# ======================================================================================

def ensure_body_cache_npz(poses_path: str, gender: str, out_npz: str):
    if osp.exists(out_npz):
        return

    tmp_dir = tempfile.gettempdir()
    exporter_py = osp.join(tmp_dir, "amass_export_body_cache_tmp.py")
    with open(exporter_py, "w", encoding="utf-8") as f:
        f.write(_EXPORTER_CODE)

    env = os.environ.copy()
    env["AMASS_DEFAULT_MOCAP_FPS"] = str(int(DEFAULT_MOCAP_FPS))

    cmd = [
        EXTERNAL_PYTHON,
        exporter_py,
        "--poses_path", poses_path,
        "--model_folder", MODEL_FOLDER,
        "--out_npz", out_npz,
        "--target_fps", str(int(TARGET_FPS)),
        "--default_gender", gender,
        "--force_gender", gender,  # always force to match your eval setting
    ]

    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(
            "Body-cache export failed.\n"
            f"CMD: {cmd}\n"
            f"STDOUT:\n{p.stdout}\n"
            f"STDERR:\n{p.stderr}\n"
        )

# ======================================================================================
# Maya scene helpers
# ======================================================================================

def _clear_scene_keep_cameras():
    default_cams = {"persp", "top", "front", "side"}
    transforms = cmds.ls(type="transform") or []
    to_delete = [t for t in transforms if t not in default_cams]
    if to_delete:
        cmds.delete(to_delete)

def _import_obj(path: str, name: str) -> str:
    if not osp.isfile(path):
        raise FileNotFoundError(path)
    before = set(cmds.ls(type="transform") or [])
    cmds.file(path, i=True, type="OBJ", ignoreVersion=True, ra=True, mergeNamespacesOnClash=False, options="mo=1")
    after = set(cmds.ls(type="transform") or [])
    new = list(after - before)
    if not new:
        # fallback: pick newest mesh transform
        meshes = cmds.ls(type="mesh") or []
        if not meshes:
            raise RuntimeError("OBJ import produced no meshes.")
        x = cmds.listRelatives(meshes[-1], parent=True, fullPath=False)[0]
    else:
        # pick transform that has mesh child
        x = None
        for t in new:
            shapes = cmds.listRelatives(t, shapes=True, type="mesh") or []
            if shapes:
                x = t
                break
        if x is None:
            x = new[0]

    x = cmds.rename(x, name)
    cmds.makeIdentity(x, apply=True, t=1, r=1, s=1, n=0)
    return x

def _create_mesh_from_verts_faces(name: str, verts, faces) -> str:
    # verts: (V,3) float32; faces: (F,3) int32
    points = [om.MPoint(float(v[0]), float(v[1]), float(v[2])) for v in verts]

    counts = om.MIntArray()
    connects = om.MIntArray()
    for tri in faces:
        counts.append(3)
        connects.append(int(tri[0]))
        connects.append(int(tri[1]))
        connects.append(int(tri[2]))

    mesh_fn = om.MFnMesh()
    created_obj = mesh_fn.create(points, counts, connects)  # may be shape OR transform depending on Maya build

    # Resolve transform robustly
    if created_obj.hasFn(om.MFn.kTransform):
        xform_obj = created_obj
    else:
        dag = om.MFnDagNode(created_obj)  # typically the SHAPE
        xform_obj = dag.parent(0)         # TRANSFORM

    xform = om.MFnDagNode(xform_obj).fullPathName()

    # Make sure target name is free (optional: delete existing)
    if cmds.objExists(name):
        # if you prefer unique names instead of delete, replace with: name = cmds.createNode("transform", name=name)
        cmds.delete(name)

    # Rename using cmds (most reliable)
    xform_name = cmds.rename(xform, name)

    # Get current shape under the renamed transform
    shapes = cmds.listRelatives(xform_name, shapes=True, fullPath=False) or []
    if shapes:
        cmds.setAttr(f"{shapes[0]}.displayColors", 0)

    cmds.makeIdentity(xform_name, apply=True, t=1, r=1, s=1, n=0)
    return xform_name

def _get_mfn_mesh(transform: str) -> om.MFnMesh:
    sel = om.MSelectionList()
    sel.add(transform)
    dag_path = sel.getDagPath(0)
    # ensure shape
    if dag_path.apiType() != om.MFn.kMesh:
        dag_path.extendToShape()
    return om.MFnMesh(dag_path)

def _set_mesh_points(transform: str, verts):
    mesh_fn = _get_mfn_mesh(transform)
    pts = om.MPointArray()
    for x, y, z in verts:
        pts.append(om.MPoint(float(x), float(y), float(z)))
    mesh_fn.setPoints(pts, om.MSpace.kObject)

def _get_mesh_points_numpy(transform: str):
    mesh_fn = _get_mfn_mesh(transform)
    pts = mesh_fn.getPoints(om.MSpace.kWorld)
    import numpy as np
    out = np.zeros((len(pts), 3), dtype=np.float32)
    for i in range(len(pts)):
        p = pts[i]
        out[i, 0], out[i, 1], out[i, 2] = float(p.x), float(p.y), float(p.z)
    return out

def _get_mesh_faces_numpy(transform: str):
    mesh_fn = _get_mfn_mesh(transform)
    counts, connects = mesh_fn.getVertices()
    # for triangles we assume mesh is triangulated (OBJ likely is)
    import numpy as np
    faces = []
    idx = 0
    for c in counts:
        if c == 3:
            faces.append([connects[idx], connects[idx+1], connects[idx+2]])
        else:
            # fan triangulate
            v0 = connects[idx]
            for k in range(1, c-1):
                faces.append([v0, connects[idx+k], connects[idx+k+1]])
        idx += c
    return np.asarray(faces, dtype=np.int32)

def _triangulate_if_needed(transform: str):
    # ensure triangles for consistent export
    shapes = cmds.listRelatives(transform, shapes=True, type="mesh") or []
    if not shapes:
        return
    cmds.select(transform, r=True)
    cmds.polyTriangulate(ch=True)
    cmds.delete(ch=True)

# ======================================================================================
# random utils
# ======================================================================================

def _assign_gray_lambert(xform: str, shader_name="bodyGray", color=(0.5, 0.5, 0.5)):
    # Create shader + SG if missing
    if not cmds.objExists(shader_name):
        shader = cmds.shadingNode("lambert", asShader=True, name=shader_name)
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shader_name + "SG")
        cmds.connectAttr(shader + ".outColor", sg + ".surfaceShader", force=True)
        cmds.setAttr(shader + ".color", color[0], color[1], color[2], type="double3")
    else:
        shader = shader_name
        sg = shader_name + "SG"

    # Assign to transform (Maya applies to shapes underneath)
    cmds.sets(xform, e=True, forceElement=sg)

def _wrap_garment_to_body(garment_xform: str, body_xform: str) -> str:
    """
    Creates a wrap deformer so garment follows body deformation.
    Must be called BEFORE creating nCloth (wrap should be upstream).
    Returns wrap node name (best-effort).
    """
    cmds.select(garment_xform, r=True)
    cmds.select(body_xform, add=True)
    mel.eval("CreateWrap;")

    wraps = cmds.ls(type="wrap") or []
    return wraps[-1] if wraps else ""

def _bake_mesh_no_history(xform: str, baked_name: str = None) -> str:
    """
    Duplicates mesh at current time, deletes history (removes wrap), returns baked transform.
    """
    baked = cmds.duplicate(xform, name=baked_name or (xform + "_baked"))[0]
    cmds.delete(baked, ch=True)
    cmds.makeIdentity(baked, apply=True, t=1, r=1, s=1, n=0)
    return baked

def _top_verts_indices_by_world_y(mesh_xform: str, top_ratio: float):
    """
    Returns vertex indices whose world-space Y is in the top top_ratio of the mesh.
    """
    import numpy as np
    pts = _get_mesh_points_numpy(mesh_xform)  # world
    y = pts[:, 1]
    y_min = float(y.min())
    y_max = float(y.max())
    if y_max <= y_min:
        return []
    thresh = y_min + (y_max - y_min) * (1.0 - float(top_ratio))
    return np.nonzero(y >= thresh)[0].tolist()

def _pin_verts_to_body_point_to_surface(garment_xform: str, body_xform: str, vert_indices):
    """
    Creates nConstraint pointToSurface so garment verts are attached to body surface (moves with body).
    Call AFTER nCloth + collider are created.
    """
    if not vert_indices:
        return ""

    cmds.select(clear=True)
    for idx in vert_indices:
        cmds.select(f"{garment_xform}.vtx[{idx}]", add=True)
    cmds.select(body_xform, add=True)

    mel.eval("createNConstraint pointToSurface 0;")
    cons = cmds.ls(type="dynamicConstraint") or []
    return cons[-1] if cons else ""

def _bbox_center_world(xform: str):
    bb = cmds.exactWorldBoundingBox(xform)  # [xmin,ymin,zmin,xmax,ymax,zmax]
    return ((bb[0]+bb[3])*0.5, (bb[1]+bb[4])*0.5, (bb[2]+bb[5])*0.5)

def _move_to_match_center(src_xform: str, dst_xform: str):
    sx, sy, sz = _bbox_center_world(src_xform)
    dx, dy, dz = _bbox_center_world(dst_xform)
    cmds.xform(src_xform, ws=True, t=(dx - sx, dy - sy, dz - sz))

def _move_to_match_center_xz(src_xform: str, dst_xform: str, y_offset: float = 0.0):
    sx, sy, sz = _bbox_center_world(src_xform)
    dx, dy, dz = _bbox_center_world(dst_xform)

    # current src translation in world
    cur = cmds.xform(src_xform, q=True, ws=True, t=True)

    # move XZ to match centers, keep current Y, then add y_offset
    new_x = cur[0] + (dx - sx)
    new_z = cur[2] + (dz - sz)
    new_y = cur[1] + float(y_offset)

    cmds.xform(src_xform, ws=True, t=(new_x, new_y, new_z))


def _axis_angle_to_mat3(r):
    # r: (3,) axis-angle
    x, y, z = float(r[0]), float(r[1]), float(r[2])
    theta = math.sqrt(x*x + y*y + z*z)
    if theta < 1e-8:
        return [[1,0,0],[0,1,0],[0,0,1]]
    ax, ay, az = x/theta, y/theta, z/theta
    c = math.cos(theta); s = math.sin(theta); C = 1.0 - c
    return [
        [c + ax*ax*C,     ax*ay*C - az*s, ax*az*C + ay*s],
        [ay*ax*C + az*s,  c + ay*ay*C,    ay*az*C - ax*s],
        [az*ax*C - ay*s,  az*ay*C + ax*s, c + az*az*C],
    ]

def _yaw_from_global_orient_axis_angle(go3):
    """
    Returns yaw in degrees (rotation about Y) inferred from SMPL global_orient.
    Assumes SMPL forward is +Z in its local frame (common convention).
    """
    R = _axis_angle_to_mat3(go3)
    # local forward (0,0,1) transformed => world forward = R * f
    fx = R[0][2]
    fz = R[2][2]
    # yaw: angle to rotate +Z toward (fx,fz) in XZ plane
    yaw_rad = math.atan2(fx, fz)
    return math.degrees(yaw_rad)


# ======================================================================================
# nCloth setup
# ======================================================================================

def _make_ncloth(garment_xform: str):
    cmds.select(garment_xform, r=True)
    mel.eval("createNCloth 0;")
    # Find created nClothShape
    cloth_shapes = cmds.ls(type="nCloth") or []
    if not cloth_shapes:
        raise RuntimeError("Failed to create nCloth.")
    cloth_shape = cloth_shapes[-1]
    return cloth_shape

def _make_collider(body_xform: str):
    cmds.select(body_xform, r=True)
    mel.eval("makeCollideNCloth;")
    rigid_shapes = cmds.ls(type="nRigid") or []
    if not rigid_shapes:
        raise RuntimeError("Failed to create nRigid collider.")
    rigid_shape = rigid_shapes[-1]
    return rigid_shape

def _get_nucleus():
    nuc = cmds.ls(type="nucleus") or []
    if not nuc:
        mel.eval("createNucleus;")
        nuc = cmds.ls(type="nucleus") or []
    return nuc[0]

def _configure_sim(nucleus: str, cloth_shape: str, rigid_shape: str, fps: int):
    # nucleus timestep
    cmds.setAttr(f"{nucleus}.gravity", GRAVITY)
    cmds.setAttr(f"{nucleus}.subSteps", int(NUCLEUS_SUBSTEPS))
    cmds.setAttr(f"{nucleus}.maxCollisionIterations", int(NUCLEUS_MAX_COLLISION_ITERS))
    # For exact FPS stepping: nucleus.spaceScale etc is tricky; keep defaults and set time step
    # Maya uses "currentUnit -time" to define fps; we set it below per-run.

    # cloth
    cmds.setAttr(f"{cloth_shape}.stretchResistance", float(CLOTH_STRETCH_RESIST))
    cmds.setAttr(f"{cloth_shape}.compressionResistance", float(CLOTH_COMPRESSION_RESIST))
    cmds.setAttr(f"{cloth_shape}.bendResistance", float(CLOTH_BEND_RESIST))
    cmds.setAttr(f"{cloth_shape}.damp", float(CLOTH_DAMP))
    cmds.setAttr(f"{cloth_shape}.thickness", float(CLOTH_THICKNESS))
    cmds.setAttr(f"{cloth_shape}.selfCollide", 1 if SELF_COLLIDE else 0)

    # collider
    cmds.setAttr(f"{rigid_shape}.thickness", float(COLLIDE_THICKNESS))

def _set_time_unit_from_fps(fps: int):
    # Maya time units: film=24, pal=25, ntsc=30, ntscf=60, etc.
    # If fps != known presets, we still simulate at ntsc and subsample; but you want 30, so use ntsc.
    if fps == 24:
        cmds.currentUnit(time="film")
    elif fps == 25:
        cmds.currentUnit(time="pal")
    elif fps == 30:
        cmds.currentUnit(time="ntsc")
    elif fps == 60:
        cmds.currentUnit(time="ntscf")
    else:
        # best effort: keep ntsc
        cmds.currentUnit(time="ntsc")

def _auto_pin_top_verts(garment_xform: str, cloth_shape: str, top_ratio: float):
    """
    Creates an nConstraint 'transform' on vertices in top top_ratio of Y (world).
    Returns list of pinned indices.
    """
    import numpy as np

    pts = _get_mesh_points_numpy(garment_xform)
    y = pts[:, 1]
    y_min = float(y.min())
    y_max = float(y.max())
    if y_max <= y_min:
        return []

    thresh = y_min + (y_max - y_min) * (1.0 - float(top_ratio))
    pinned = np.nonzero(y >= thresh)[0].tolist()
    if not pinned:
        return []

    # select vertices
    cmds.select(clear=True)
    for idx in pinned:
        cmds.select(f"{garment_xform}.vtx[{idx}]", add=True)

    # Create constraint. This makes a transform constraint to world by default.
    mel.eval("createNConstraint transform 0;")

    # Make pinned points very heavy (optional): nCloth has pointMass but per-vertex is complex; constraint is enough.
    return pinned

# ======================================================================================
# PLY export (binary_little_endian)
# ======================================================================================

_SCALAR_FMT = {"float": ("f", 4), "int": ("i", 4), "uint": ("I", 4), "uchar": ("B", 1)}

def _write_ply_binary_le(path: str, verts, faces):
    """
    verts: (V,3) float32/float
    faces: (F,3) int
    """
    _ensure_dir(osp.dirname(path))
    V = len(verts)
    F = len(faces)

    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        f"element vertex {V}\n"
        "property float x\n"
        "property float y\n"
        "property float z\n"
        f"element face {F}\n"
        "property list uchar int vertex_indices\n"
        "end_header\n"
    ).encode("ascii")

    with open(path, "wb") as f:
        f.write(header)
        for v in verts:
            f.write(struct.pack("<fff", float(v[0]), float(v[1]), float(v[2])))
        for tri in faces:
            f.write(struct.pack("<Biii", 3, int(tri[0]), int(tri[1]), int(tri[2])))

# ======================================================================================
# Optional render: playblast -> mp4
# ======================================================================================

def _playblast_to_images(img_dir: str, start: int, end: int, fps: int, pad: int = 4):
    _rm_tree_if_exists(img_dir)
    _ensure_dir(img_dir)

    cmds.playbackOptions(min=start, max=end)
    cmds.currentTime(start)

    # Use playblast to image sequence
    out_path_noext = osp.join(img_dir, "frame_")
    cmds.playblast(
        format="image",
        filename=out_path_noext,
        framePadding=pad,
        compression="png",
        startTime=start,
        endTime=end,
        sequenceTime=True,
        clearCache=True,
        viewer=False,
        showOrnaments=False,
        offScreen=True,
        percent=100,
        widthHeight=(PLAYBLAST_W, PLAYBLAST_H),
        forceOverwrite=True,
    )

def _images_to_mp4(img_dir: str, out_mp4: str, fps: int, pad: int = 4):
    # Maya's playblast names often end like frame_0000.png
    pattern = osp.join(img_dir, f"frame_%0{pad}d.png")
    cmd = [
        FFMPEG_BIN,
        "-y",
        "-framerate", str(int(fps)),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        out_mp4,
    ]
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")

# ======================================================================================
# Main pipeline: process_example
# ======================================================================================

def process_example(sequence_num: str, sequence_idx: str, garment: str, gender: str):
    """
    Call this in your loop.

    Inputs:
      sequence_num: "01", "02", ...
      sequence_idx: "01", "02", ...
      garment: "t-shirt" / "shirt" / "pant" / ...
      gender: "male" / "female"

    Outputs:
      OUT_ROOT/CMU/<num>/<idx>/<gender>/<garment>/{body,cloth}/... ply
      optionally mp4 output_N.mp4 in that same folder
    """
    sequence_num = str(sequence_num)
    sequence_idx = str(sequence_idx)
    garment = str(garment)
    gender = str(gender)

    poses_path = find_amass_poses_file(AMASS_ROOT, sequence_num, sequence_idx)
    garment_obj = find_garment_obj(GARMENT_OBJ_DIR, garment, gender, sequence_num)

    out_dir = osp.join(OUT_ROOT, "CMU", sequence_num, sequence_idx, gender, garment)
    body_dir = osp.join(out_dir, "body")
    cloth_dir = osp.join(out_dir, "cloth")
    _ensure_dir(body_dir)
    _ensure_dir(cloth_dir)

    body_cache = osp.join(out_dir, "body_cache.npz")
    ensure_body_cache_npz(poses_path, gender, body_cache)

    # Load cache
    import numpy as np
    cache = np.load(body_cache, allow_pickle=True)
    verts_seq = cache["verts"]           # (T,V,3)
    faces = cache["faces"]              # (F,3)
    fps = int(cache.get("fps", TARGET_FPS))
    T = int(verts_seq.shape[0])

    if T < 3:
        raise RuntimeError(f"Sequence too short after subsample: T={T}")

    if CLEAR_SCENE_EACH_RUN:
        _clear_scene_keep_cameras()

    _set_time_unit_from_fps(fps)

    # 1) Create body mesh (frame 0)
    body_xform = _create_mesh_from_verts_faces("amass_body", verts_seq[0], faces)
    _assign_gray_lambert(body_xform, color=(0.35, 0.35, 0.35))
    cmds.rotate(-90, 0, 0, body_xform, r=True, os=True, fo=True)
    _triangulate_if_needed(body_xform)


    y_offset = 1.1

    # Timeline early
    start = 0
    end = T - 1
    cmds.playbackOptions(min=start, max=end)

    # Make sure time is at start and body is posed for frame 0
    cmds.currentTime(start, edit=True)
    _set_mesh_points(body_xform, verts_seq[start])
    cmds.refresh(f=True)

    # 2) Import garment OBJ
    garment_xform = _import_obj(garment_obj, f"garment_{garment}_{gender}_{sequence_num}")
    _triangulate_if_needed(garment_xform)
    # _move_to_match_center(garment_xform, body_xform)
    _move_to_match_center_xz(garment_xform, body_xform, y_offset)
    go_seq = cache["global_orient"]  # (T,3)
    yaw0 = _yaw_from_global_orient_axis_angle(go_seq[0])
    cmds.rotate(0, yaw0, 0, garment_xform, r=True, os=True, fo=True)
    cmds.makeIdentity(garment_xform, apply=True, t=0, r=1, s=0, n=0)
    cmds.refresh(f=True)

    # 2.5) WRAP DRESSING (before nCloth!)
    wrap_node = _wrap_garment_to_body(garment_xform, body_xform)
    cmds.refresh(f=True)

    if wrap_node:
        if cmds.attributeQuery("maxDistance", node=wrap_node, exists=True):
            cmds.setAttr(wrap_node + ".maxDistance", 1e9)
        if cmds.attributeQuery("weightThreshold", node=wrap_node, exists=True):
            cmds.setAttr(wrap_node + ".weightThreshold", 0.0)

        # These attrs exist in many Maya versions; set if present
        if cmds.attributeQuery("exclusiveBind", node=wrap_node, exists=True):
            cmds.setAttr(wrap_node + ".exclusiveBind", 0)
        if cmds.attributeQuery("autoWeightThreshold", node=wrap_node, exists=True):
            cmds.setAttr(wrap_node + ".autoWeightThreshold", 0)

    if not USE_WRAP_THROUGHOUT:
        baked = _bake_mesh_no_history(garment_xform, baked_name=garment_xform + "_sim")
        cmds.delete(garment_xform)
        garment_xform = baked
        _triangulate_if_needed(garment_xform)

    # 3) nCloth setup (on the FINAL garment_xform)
    cloth_shape = _make_ncloth(garment_xform)
    rigid_shape = _make_collider(body_xform)
    nucleus = _get_nucleus()
    _configure_sim(nucleus, cloth_shape, rigid_shape, fps)

    # make solver start consistent with the timeline
    cmds.setAttr(f"{nucleus}.startFrame", int(start))

    # 4) Pin top verts to BODY (after nCloth + collider exist)
    pinned = _top_verts_indices_by_world_y(garment_xform, AUTO_PIN_TOP_RATIO)
    _pin_verts_to_body_point_to_surface(garment_xform, body_xform, pinned)

    # Ensure scene eval starts clean (optional but ok)
    cmds.currentTime(start, edit=True); cmds.refresh(f=True)
    cmds.currentTime(start+1, edit=True); cmds.refresh(f=True)
    cmds.currentTime(start, edit=True); cmds.refresh(f=True)





    # Cache faces for export (use triangulated Maya meshes to avoid mismatch)
    gar_faces = _get_mesh_faces_numpy(garment_xform)
    body_faces = _get_mesh_faces_numpy(body_xform)

    for fidx in range(start+1, T):
        # Update body mesh points for this frame (object space)
        _set_mesh_points(body_xform, verts_seq[fidx])

        cmds.currentTime(fidx, edit=True)

        # Force DG evaluation so cloth sees updated collider
        cmds.refresh(f=True)

        # Export
        b_verts = _get_mesh_points_numpy(body_xform)
        g_verts = _get_mesh_points_numpy(garment_xform)

        idx_str = str(fidx).zfill(EXPORT_PAD)
        body_ply = osp.join(body_dir, f"body_{idx_str}.ply")
        cloth_ply = osp.join(cloth_dir, f"pred_gar_{idx_str}.ply")

        _write_ply_binary_le(body_ply, b_verts, body_faces)
        _write_ply_binary_le(cloth_ply, g_verts, gar_faces)

        if (fidx % 25) == 0:
            print(f"[{sequence_num}_{sequence_idx} {garment} {gender}] frame {fidx}/{T-1}")

    print(f"[DONE] Exported {T} frames to:\n  {body_dir}\n  {cloth_dir}")
    print(f"[INFO] Pinned verts: {len(pinned)} (top {AUTO_PIN_TOP_RATIO*100:.1f}% Y)")

    # Optional mp4
    if MAKE_MP4:
        img_dir = osp.join(out_dir, "_image_renders")
        mp4_out = _next_available_mp4(out_dir, base="output", ext=".mp4")
        _playblast_to_images(img_dir, start, end, fps=fps, pad=EXPORT_PAD)
        _images_to_mp4(img_dir, mp4_out, fps=fps, pad=EXPORT_PAD)
        print("[MP4]", mp4_out)

    return out_dir


# ======================================================================================
# Example loop (your pattern)
# ======================================================================================

if __name__ == "__main__":
    sequences = ["01", "02", "05", "07"]
    sequences = ["07"]
    sequence_indices = ["01", "02", "03", "04", "05"]
    garments = ["t-shirt", "shirt", "pant"]

    for seq_num in sequences:
        for seq_idx in sequence_indices:
            for garment in garments:
                try:
                    process_example(seq_num, seq_idx, garment, "male")
                except Exception as e:
                    print(f"[FAILED] {seq_num}_{seq_idx} {garment} male: {e}")
                break
            break
        break
