# Maya ground-truth cloth sim for AMASS (CMU) + TailorNet garment OBJs
# VERSION: Final Tuned (Exposed Constants + Structural Physics)

from maya.api import OpenMaya as om
import maya.cmds as cmds
import maya.mel as mel

import os
import os.path as osp
import tempfile
import subprocess
import struct
import numpy as np

# ======================================================================================
# CONFIGURATION
# ======================================================================================

AMASS_ROOT = r"D:\ClothSim\AMASS"
MODEL_FOLDER = r"D:\ClothSim" 
EXTERNAL_PYTHON = r"C:\Users\janr\Documents\MagCode\.venv_py310\Scripts\python.exe"
GARMENT_OBJ_DIR = r"D:\ClothSim\ccraft_data\aux_data\garment_meshes\smpl"
OUT_ROOT = r"D:\ClothSim\Results\Maya"

TARGET_FPS = 30
DEFAULT_MOCAP_FPS = 30
CLEAR_SCENE_EACH_RUN = True
EXPORT_PAD = 4
AUTO_PIN_TOP_RATIO = 0.02

# edit based on the parameters
ISSILK = True
ISDENIM = False

# --- PHYSICS PARAMETERS (HEAVY COTTON SETTINGS) ---
GRAVITY = 9.8
NUCLEUS_SUBSTEPS = 15             # High substeps essential for stiff constraints
NUCLEUS_MAX_COLLISION_ITERS = 30

# Mass
CLOTH_MASS = 0.25                 # kg/m^2 (Slightly heavier than 0.2 for stability)

# Structure (Stiffness)
CLOTH_STRETCH_RESIST = 800.0      # Locks edge length
CLOTH_COMPRESSION_RESIST = 250
CLOTH_SHEAR_RESIST = 800          # Prevents "liquid" skewing
CLOTH_DEFORM_RESIST = 0.0        # KEY: Holds local shape structure (stops fluid look)
CLOTH_BEND_RESIST = 1.0           # KEY: Higher value = larger, smoother folds (Cotton)
CLOTH_BEND_ANGLE_DROPOFF = 0.6    

# Damping / Drag
CLOTH_DAMP = 0.8                  # Low to allow motion, high enough to stop jitter
CLOTH_DRAG = 0.05                  # 0.0 prevents "lagging" behind character
CLOTH_TANGENTIAL_DRAG = 0.0

# Friction / Contact
CLOTH_FRICTION = 0.6
CLOTH_STICKINESS = 0.1            # Helps keep shirt on shoulders
CLOTH_THICKNESS = 0.005           # 3mm visual thickness
COLLIDE_THICKNESS = 0.005
CLOTH_SELF_COLLIDE_WIDTH_SCALE = 1.5 # Padding multiplier for self-collision

# Collider (Body)
COLLIDER_THICKNESS = 0.005
COLLIDER_FRICTION = 0.6
COLLIDER_STICKINESS = 0.1

if ISSILK:
    CLOTH_MASS = 0.1
    CLOTH_STRETCH_RESIST = 800.0
    CLOTH_COMPRESSION_RESIST = 20.0
    CLOTH_BEND_RESIST = 0.05
    CLOTH_DAMP = 0.2
    CLOTH_FRICTION = 0.1

if ISDENIM:
    CLOTH_MASS = 1.5
    CLOTH_STRETCH_RESIST = 1000.0
    CLOTH_COMPRESSION_RESIST = 800.0
    CLOTH_BEND_RESIST = 15.0
    CLOTH_DAMP = 1.0
    CLOTH_FRICTION = 0.8
    CLOTH_THICKNESS = 0.006
    COLLIDE_THICKNESS = 0.006
    COLLIDER_THICKNESS = 0.006

# ======================================================================================
# HELPER: SAFE ATTRIBUTE SETTER
# ======================================================================================
def set_attr_safe(node, attr, value):
    """
    Sets an attribute by selecting the node first. 
    This bypasses errors where setAttr fails to resolve the DAG path string.
    """
    try:
        # 1. Select the node (Force Maya to acknowledge it)
        cmds.select(node)
        # 2. Set attribute on "selected" (represented by dot)
        cmds.setAttr(f".{attr}", value)
    except Exception as e:
        print(f"[WARNING] Failed to set {node}.{attr} to {value}: {e}")
        # Try one last desperate attempt with full string
        try:
            cmds.setAttr(f"{node}.{attr}", value)
        except:
            pass

def safe_index(lst, idx=0):
    if lst and len(lst) > idx: return lst[idx]
    return None

# ======================================================================================
# EXTERNAL EXPORTER CODE
# ======================================================================================

_EXPORTER_CODE = r"""
import os
import os.path as osp
import numpy as np

if not hasattr(np, "bool"): np.bool = bool
if not hasattr(np, "int"): np.int = int
if not hasattr(np, "float"): np.float = float
if not hasattr(np, "complex"): np.complex = complex
if not hasattr(np, "object"): np.object = object
if not hasattr(np, "unicode"): np.unicode = str
if not hasattr(np, "str"): np.str = str

import torch
import smplx

DEFAULT_MOCAP_FPS = int(os.environ.get("AMASS_DEFAULT_MOCAP_FPS", "30"))

def load_amass_raw(poses_path: str):
    if not osp.isfile(poses_path): raise FileNotFoundError(poses_path)
    return dict(np.load(poses_path, allow_pickle=True))

def load_shape(seq_dir: str, raw: dict, force_gender=None, default_gender="female"):
    betas = raw.get("betas", None)
    gender = raw.get("gender", None)
    shp = osp.join(seq_dir, "shape.npz")
    if osp.exists(shp):
        s = dict(np.load(shp, allow_pickle=True))
        if "betas" in s: betas = s["betas"]
        if "gender" in s and gender is None: gender = s["gender"]
    if betas is None: betas = np.zeros(16, dtype=np.float32)
    betas = np.asarray(betas, dtype=np.float32).reshape(-1)
    if force_gender is not None: gender = force_gender
    if gender is None: gender = default_gender
    else: gender = str(gender)
    return betas, gender

def parse_smpl_params(raw: dict):
    poses = np.asarray(raw["poses"], dtype=np.float32)
    global_orient = poses[:, :3]
    body_pose = poses[:, 3:72]
    trans = raw.get("trans", None)
    if trans is None: trans = np.zeros((poses.shape[0], 3), dtype=np.float32)
    trans = np.asarray(trans, dtype=np.float32)
    fps = raw.get("mocap_framerate", raw.get("mocap_fps", DEFAULT_MOCAP_FPS))
    fps = int(np.array(fps).item())
    return global_orient, body_pose, trans, fps

def export_body_cache(poses_path, model_folder, out_npz, target_fps=30, force_gender=None, default_gender="female"):
    raw = load_amass_raw(poses_path)
    seq_dir = osp.dirname(poses_path)
    betas, gender = load_shape(seq_dir, raw, force_gender=force_gender, default_gender=default_gender)
    go, bp, tr, mocap_fps = parse_smpl_params(raw)
    skip = int(round(float(mocap_fps) / float(target_fps)))
    skip = max(skip, 1)
    go = go[::skip]; bp = bp[::skip]; tr = tr[::skip]
    T = go.shape[0]

    model = smplx.create(model_path=model_folder, model_type="smpl", gender=gender, use_pca=False, batch_size=T)
    betas_t = np.tile(betas[None, :], (T, 1)).astype(np.float32)[:, :model.num_betas]

    with torch.no_grad():
        out = model(betas=torch.from_numpy(betas_t).float(), body_pose=torch.from_numpy(bp).float(),
                    global_orient=torch.from_numpy(go).float(), transl=torch.from_numpy(tr).float(), return_verts=True)
    verts = out.vertices.detach().cpu().numpy().astype(np.float32)
    faces = model.faces.astype(np.int32)

    model_rest = smplx.create(model_path=model_folder, model_type="smpl", gender=gender, batch_size=1)
    zeros_pose = torch.zeros((1, 69), dtype=torch.float32)
    zeros_orient = torch.zeros((1, 3), dtype=torch.float32)
    zeros_trans = torch.zeros((1, 3), dtype=torch.float32)
    betas_rest = torch.from_numpy(betas[:model.num_betas][None, :]).float()
    with torch.no_grad():
        out_rest = model_rest(betas=betas_rest, body_pose=zeros_pose, global_orient=zeros_orient, 
                              transl=zeros_trans, return_verts=True)
    rest_verts = out_rest.vertices.detach().cpu().numpy().astype(np.float32)[0]

    os.makedirs(osp.dirname(out_npz), exist_ok=True)
    np.savez_compressed(out_npz, verts=verts, faces=faces, rest_verts=rest_verts, 
                        fps=int(target_fps), skip=int(skip), gender=str(gender))
    print("Wrote:", out_npz)

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
# MAYA HELPERS
# ======================================================================================

def _ensure_dir(p: str):
    if not osp.exists(p): os.makedirs(p)

def _get_mfn_mesh(transform: str) -> om.MFnMesh:
    sel = om.MSelectionList()
    sel.add(transform)
    dag_path = sel.getDagPath(0)
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
    out = np.zeros((len(pts), 3), dtype=np.float32)
    for i in range(len(pts)):
        out[i] = [pts[i].x, pts[i].y, pts[i].z]
    return out

def _create_mesh_from_verts_faces(name: str, verts, faces) -> str:
    points = [om.MPoint(float(v[0]), float(v[1]), float(v[2])) for v in verts]
    counts = om.MIntArray([3] * len(faces))
    connects = om.MIntArray()
    for tri in faces:
        connects.append(int(tri[0])); connects.append(int(tri[1])); connects.append(int(tri[2]))
    mesh_fn = om.MFnMesh()
    created_obj = mesh_fn.create(points, counts, connects)
    if created_obj.hasFn(om.MFn.kTransform): xform_obj = created_obj
    else: xform_obj = om.MFnDagNode(created_obj).parent(0)
    xform = om.MFnDagNode(xform_obj).fullPathName()
    if cmds.objExists(name): cmds.delete(name)
    xform_name = cmds.rename(xform, name)
    shapes = cmds.listRelatives(xform_name, shapes=True)
    if shapes: cmds.setAttr(f"{shapes[0]}.displayColors", 0)
    cmds.makeIdentity(xform_name, apply=True, t=1, r=1, s=1, n=0)
    return xform_name

def _import_obj(path: str, name: str) -> str:
    before = set(cmds.ls(type="transform") or [])
    cmds.file(path, i=True, type="OBJ", ignoreVersion=True, options="mo=1")
    after = set(cmds.ls(type="transform") or [])
    new = list(after - before)
    if not new: raise RuntimeError(f"Failed to import {path}")
    x = new[0]
    for t in new:
        if cmds.listRelatives(t, shapes=True, type="mesh"):
            x = t; break
    if cmds.objExists(name): cmds.delete(name)
    x = cmds.rename(x, name)
    cmds.makeIdentity(x, apply=True, t=1, r=1, s=1, n=0)
    return x

def _assign_gray_lambert(xform: str, shader_name="bodyGray", color=(0.35, 0.35, 0.35)):
    if not cmds.objExists(shader_name):
        shader = cmds.shadingNode("lambert", asShader=True, name=shader_name)
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=shader_name + "SG")
        cmds.connectAttr(shader + ".outColor", sg + ".surfaceShader", force=True)
        cmds.setAttr(shader + ".color", *color, type="double3")
    else: sg = shader_name + "SG"
    cmds.sets(xform, e=True, forceElement=sg)

def _clear_scene_keep_cameras():
    all_roots = cmds.ls(assemblies=True)
    defaults = {"persp", "top", "front", "side"}
    to_delete = [x for x in all_roots if x not in defaults]
    if to_delete: cmds.delete(to_delete)
    sim_types = ["nucleus", "nCloth", "nRigid", "dynamicConstraint", "wrap"]
    for t in sim_types:
        nodes = cmds.ls(type=t)
        if nodes:
            try: cmds.delete(nodes)
            except: pass

def set_ncloth_uniform_mass(cloth_shape: str, cloth_mesh_xform: str, mass_value: float):
    n = int(cmds.polyEvaluate(cloth_mesh_xform, vertex=True))
    full_attr = cloth_shape + ".mass"
    vals = [float(mass_value)] * n
    try:
        cmds.setAttr(full_attr, vals, type="doubleArray")
    except Exception as e:
        print(f"[ERROR] Failed to set per-vertex mass: {e}")

def _configure_sim(nucleus, cloth_shape, rigid_shape, cloth_mesh_xform):    
    # === SOLVER ===
    set_attr_safe(nucleus, "spaceScale", 1.0)
    set_attr_safe(nucleus, "gravity", GRAVITY)
    set_attr_safe(nucleus, "subSteps", NUCLEUS_SUBSTEPS)
    set_attr_safe(nucleus, "maxCollisionIterations", NUCLEUS_MAX_COLLISION_ITERS)

    # === CLOTH ===
    set_ncloth_uniform_mass(cloth_shape, cloth_mesh_xform, CLOTH_MASS)
    
    set_attr_safe(cloth_shape, "stretchResistance", CLOTH_STRETCH_RESIST)
    set_attr_safe(cloth_shape, "compressionResistance", CLOTH_COMPRESSION_RESIST)
    set_attr_safe(cloth_shape, "shearResistance", CLOTH_SHEAR_RESIST)
    
    # Bending & Deform (Structure)
    set_attr_safe(cloth_shape, "bendResistance", CLOTH_BEND_RESIST)
    set_attr_safe(cloth_shape, "bendAngleDropoff", CLOTH_BEND_ANGLE_DROPOFF)
    set_attr_safe(cloth_shape, "deformResistance", CLOTH_DEFORM_RESIST)
    
    # Friction / Damp / Drag
    set_attr_safe(cloth_shape, "friction", CLOTH_FRICTION)
    set_attr_safe(cloth_shape, "stickiness", CLOTH_STICKINESS)
    set_attr_safe(cloth_shape, "damp", CLOTH_DAMP)
    set_attr_safe(cloth_shape, "tangentialDrag", CLOTH_TANGENTIAL_DRAG)
    set_attr_safe(cloth_shape, "drag", CLOTH_DRAG)
    
    # Collisions
    set_attr_safe(cloth_shape, "selfCollide", 1)
    set_attr_safe(cloth_shape, "selfCollisionFlag", 1)
    set_attr_safe(cloth_shape, "thickness", CLOTH_THICKNESS)
    set_attr_safe(cloth_shape, "selfCollideWidthScale", CLOTH_SELF_COLLIDE_WIDTH_SCALE)
    
    # === COLLIDER ===
    set_attr_safe(rigid_shape, "thickness", COLLIDE_THICKNESS)
    set_attr_safe(rigid_shape, "friction", COLLIDER_FRICTION)
    set_attr_safe(rigid_shape, "stickiness", COLLIDER_STICKINESS)

# ======================================================================================
# LOGIC
# ======================================================================================

def _fit_garment_via_wrap(garment_tpose_xform, body_tpose_verts, body_target_verts, faces):
    temp_driver = _create_mesh_from_verts_faces("temp_driver_body", body_tpose_verts, faces)
    cmds.rotate(-90, 0, 0, temp_driver, r=True)
    
    cmds.select(clear=True)
    cmds.select(garment_tpose_xform, r=True)
    cmds.select(temp_driver, add=True)
    mel.eval("CreateWrap;")
    
    wraps = cmds.ls(type="wrap") or []
    if wraps:
        w = wraps[-1]
        cmds.setAttr(f"{w}.exclusiveBind", 1)
        cmds.setAttr(f"{w}.autoWeightThreshold", 1)
        cmds.setAttr(f"{w}.falloffMode", 0)
    
    _set_mesh_points(temp_driver, body_target_verts)
    cmds.dgdirty(allPlugs=True)
    cmds.refresh(f=True)
    
    baked_name = garment_tpose_xform + "_fitted"
    baked = cmds.duplicate(garment_tpose_xform, name=baked_name)[0]
    
    cmds.delete(temp_driver)
    if cmds.objExists(garment_tpose_xform): cmds.delete(garment_tpose_xform)
    cmds.delete(baked, ch=True) 
    return baked

def ensure_body_cache_npz(poses_path, gender, out_npz):
    if osp.exists(out_npz):
        try:
            d = np.load(out_npz, allow_pickle=True)
            if "rest_verts" in d: return
        except: pass

    tmp_dir = tempfile.gettempdir()
    exporter_py = osp.join(tmp_dir, "amass_exporter.py")
    with open(exporter_py, "w") as f: f.write(_EXPORTER_CODE)
    
    env = os.environ.copy()
    env["AMASS_DEFAULT_MOCAP_FPS"] = str(int(DEFAULT_MOCAP_FPS))
    
    cmd = [EXTERNAL_PYTHON, exporter_py, "--poses_path", poses_path, "--model_folder", MODEL_FOLDER, 
           "--out_npz", out_npz, "--target_fps", str(TARGET_FPS), "--force_gender", gender]
           
    p = subprocess.run(cmd, capture_output=True, text=True, env=env)
    if p.returncode != 0:
        raise RuntimeError(f"Export failed:\n{p.stderr}")

def process_example(sequence_num, sequence_idx, garment, gender):
    sequence_num, sequence_idx = str(sequence_num), str(sequence_idx)
    poses_path = osp.join(AMASS_ROOT, "CMU", sequence_num, f"{sequence_num}_{sequence_idx}_poses.npz")
    garment_obj = osp.join(GARMENT_OBJ_DIR, f"tailornet_{garment}_{gender}_{sequence_num}.obj")
    
    out_dir = osp.join(OUT_ROOT, "CMU", sequence_num, sequence_idx, gender, garment, "result_ply_files")
    body_cache = osp.join(out_dir, "body_cache.npz")
    
    ensure_body_cache_npz(poses_path, gender, body_cache)
    
    cache = np.load(body_cache, allow_pickle=True)
    verts_seq, faces = cache["verts"], cache["faces"]
    rest_verts = cache["rest_verts"]
    
    if CLEAR_SCENE_EACH_RUN:
        _clear_scene_keep_cameras()
    
    cmds.currentUnit(time="ntsc")
    cmds.playbackOptions(min=0, max=len(verts_seq)-1, playbackSpeed=0)
    
    # 1. Garment & Fit
    garment_xform = _import_obj(garment_obj, "garment_mesh")
    cmds.rotate(-90, 0, 0, garment_xform, r=True)
    garment_xform = _fit_garment_via_wrap(garment_xform, rest_verts, verts_seq[0], faces)
    
    # 2. Body
    body_xform = _create_mesh_from_verts_faces("body_mesh", verts_seq[0], faces)
    cmds.rotate(-90, 0, 0, body_xform, r=True)
    _assign_gray_lambert(body_xform)

    # 3. Create nCloth
    existing_cloths = set(cmds.ls(type="nCloth", long=True) or [])
    cmds.select(garment_xform)
    mel.eval("createNCloth 0;")
    new_cloths = set(cmds.ls(type="nCloth", long=True) or []) - existing_cloths
    
    if not new_cloths: cloth_shape = safe_index(cmds.ls(type="nCloth", long=True), -1)
    else: cloth_shape = list(new_cloths)[0]

    if not cloth_shape: raise RuntimeError("nCloth creation failed.")
    print(f"[INFO] nCloth Shape: {cloth_shape}")

    # Detect New Output Mesh
    out_mesh_shapes = cmds.listConnections(cloth_shape + ".outputMesh", type="mesh")
    if out_mesh_shapes:
        parents = cmds.listRelatives(out_mesh_shapes[0], parent=True, fullPath=True)
        if parents:
            out_transform = parents[0]
            garment_full = cmds.ls(garment_xform, long=True)[0]
            if out_transform != garment_full:
                print(f"[INFO] Swapping export target to: {out_transform}")
                try: cmds.setAttr(f"{garment_xform}.visibility", 0)
                except: pass
                garment_xform = out_transform

    # 4. Create Collider
    existing_rigids = set(cmds.ls(type="nRigid", long=True) or [])
    cmds.select(body_xform)
    mel.eval("makeCollideNCloth;")
    new_rigids = set(cmds.ls(type="nRigid", long=True) or []) - existing_rigids
    
    if not new_rigids:
        hist = cmds.listHistory(body_xform) or []
        found = [h for h in hist if cmds.nodeType(h) == "nRigid"]
        rigid_shape = cmds.ls(found[0], long=True)[0] if found else safe_index(cmds.ls(type="nRigid", long=True), -1)
    else:
        rigid_shape = list(new_rigids)[0]
        
    if not rigid_shape: raise RuntimeError("nRigid creation failed.")

    nucleus = safe_index(cmds.ls(type="nucleus", long=True), 0)
    
    # 5. Physics
    _configure_sim(nucleus, cloth_shape, rigid_shape, garment_xform)
    set_attr_safe(nucleus, "startFrame", 0)
    
    # 6. Pinning
    pts = _get_mesh_points_numpy(garment_xform)
    y_vals = pts[:, 1]
    thresh = np.min(y_vals) + (np.max(y_vals) - np.min(y_vals)) * (1.0 - AUTO_PIN_TOP_RATIO)
    pinned_indices = np.where(y_vals >= thresh)[0]
    
    if len(pinned_indices) > 0:
        cmds.select(clear=True)
        for idx in pinned_indices: cmds.select(f"{garment_xform}.vtx[{idx}]", add=True)
        cmds.select(body_xform, add=True)
        mel.eval("createNConstraint pointToSurface 0;")
    
    # 7. Loop
    def write_ply(fname, v, f):
        header = f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\nproperty float x\nproperty float y\nproperty float z\nelement face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n"
        with open(fname, "wb") as file:
            file.write(header.encode('ascii'))
            file.write(struct.pack(f"<{len(v)*3}f", *v.flatten()))
            for tri in f: file.write(struct.pack("<Biii", 3, tri[0], tri[1], tri[2]))

    gar_faces = np.array(_get_mfn_mesh(garment_xform).getVertices()[1], dtype=np.int32).reshape(-1, 3)

    print(f"Starting Sim: {sequence_num}_{sequence_idx}...")
    
    # initialize frame 0
    cmds.currentTime(0)
    _set_mesh_points(body_xform, verts_seq[0])
    for frame in range(len(verts_seq)):
        _set_mesh_points(body_xform, verts_seq[frame])
        cmds.currentTime(frame)
        cmds.dgdirty(allPlugs=True)
        
        g_v = _get_mesh_points_numpy(garment_xform)
        b_v = _get_mesh_points_numpy(body_xform)
        
        idx = str(frame).zfill(EXPORT_PAD)
        write_ply(osp.join(out_dir, f"body_{idx}.ply"), b_v, faces)
        write_ply(osp.join(out_dir, f"pred_gar_{idx}.ply"), g_v, gar_faces)
        if frame % 10 == 0:
            print(f"Frame {frame}/{len(verts_seq)}")
        # if frame == 30:
        #     break

    print(f"DONE: {out_dir}")




if __name__ == "__main__":
    try:
        process_example("07", "01", "t-shirt", "male")
    except Exception as e:
        import traceback
        traceback.print_exc()

    # ------------------------------------------------------ #

    # sequences = ["01", "02", "05", "07"]
    # sequence_indices = ["01", "02", "03", "04", "05"]
    # garments  = ["t-shirt", "shirt", "pant"]
    # gender = "male"

    # for seq_num in sequences:
    #     for seq_idx in sequence_indices:
    #         for garment in garments:
    #             try:
    #                 process_example(seq_num, seq_idx, garment, gender)
    #             except Exception as e:
    #                 print(f"[FAILED] {seq_num}_{seq_idx} {garment} {gender}: {e}")