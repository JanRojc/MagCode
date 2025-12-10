import os
import os.path as osp
import sys
import traceback

import numpy as np
import maya.cmds as cmds
import maya.api.OpenMaya as om

# -------------------------------------------------------------------------
# 1. Point this to your TailorNet_dataset *code* repo
#    (the one that has global_var.py, smpl_torch.py, visualize_dataset.py)
# -------------------------------------------------------------------------
TN_DATASET_REPO = "/Users/jan.rojc/Documents/MagCode/RandomRepos/TailorNet_dataset_repo"
TN_DATASET = "/Users/jan.rojc/Documents/MagCode/Data/TailorNet"
if TN_DATASET_REPO not in sys.path:
    sys.path.append(TN_DATASET_REPO)

from smpl_torch import SMPLNP               # TailorNet SMPL wrapper


# -------------------------------------------------------------------------
# Small helpers
# -------------------------------------------------------------------------

def _clear_scene_keep_cameras():
    """Delete all transforms except default cameras."""
    default_cams = {"persp", "top", "front", "side"}
    transforms = cmds.ls(type="transform") or []
    to_delete = [t for t in transforms if t not in default_cams]
    if to_delete:
        cmds.delete(to_delete)


def _create_mesh_in_maya(verts, faces, name="tailornet_garment"):
    """
    Create a polygon mesh in Maya from numpy arrays.

    verts: (N, 3) float
    faces: (F, 3) int
    """
    points = om.MFloatPointArray()
    for v in verts:
        points.append(om.MFloatPoint(float(v[0]), float(v[1]), float(v[2])))

    face_counts = om.MIntArray()
    face_connects = om.MIntArray()
    for f in faces:
        face_counts.append(3)
        face_connects.append(int(f[0]))
        face_connects.append(int(f[1]))
        face_connects.append(int(f[2]))

    mesh_fn = om.MFnMesh()
    new_mesh = mesh_fn.create(points, face_counts, face_connects)
    transform = mesh_fn.parent(0)
    dag = om.MFnDagNode(transform)
    transform_name = cmds.rename(dag.name(), name)

    # JanR
    # recolor the mesh to gray
    mat = cmds.shadingNode("lambert", asShader=True, name=name + "_mat")
    cmds.setAttr(mat + ".color", 0.5, 0.5, 0.5, type="double3")
    sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True,
                   name=name + "_SG")
    cmds.connectAttr(mat + ".outColor", sg + ".surfaceShader", force=True)
    cmds.sets(transform_name, e=True, forceElement=sg)

    return transform_name


# -------------------------------------------------------------------------
# 2. TailorNet → Maya: reconstruct proper gar_v via SMPLNP
# -------------------------------------------------------------------------

def _load_garment_faces(garment_class):
    """
    Load garment template faces from TailorNet dataset_meta/garment_class_info.pkl
    """
    meta_path = osp.join(TN_DATASET, "dataset_meta", "garment_class_info.pkl")
    if not osp.exists(meta_path):
        cmds.error("garment_class_info.pkl not found at: {}".format(meta_path))
        return None

    import pickle
    with open(meta_path, "rb") as f:
        info = pickle.load(f, encoding="latin1")

    if garment_class not in info:
        cmds.error("Garment class '{}' not found in garment_class_info.pkl".format(
            garment_class
        ))
        return None

    faces = np.asarray(info[garment_class]["f"], dtype=np.int32)  # (F, 3)
    if faces.ndim != 2 or faces.shape[1] != 3:
        cmds.error("Unexpected faces shape for {}: {}".format(
            garment_class, faces.shape
        ))
        return None
    return faces


def _find_style_shape_file(class_dir, beta_str, gamma_str):
    """
    Robustly find style_shape file for given (beta_str, gamma_str).
    TailorNet uses names like:
      'beta{}_gamma{}.npy' OR 'beta {}_gamma {}.npy'
    depending on version.
    """
    ss_dir = osp.join(class_dir, "style_shape")
    candidates = [
        f"beta{beta_str}_gamma{gamma_str}.npy",
        f"beta_{beta_str}_gamma_{gamma_str}.npy",
        f"beta {beta_str}_gamma {gamma_str}.npy",
    ]
    for fn in candidates:
        path = osp.join(ss_dir, fn)
        if osp.exists(path):
            return path
    cmds.error(
        "Could not find style_shape file for beta={}, gamma={} in {}.\nTried:\n  {}".format(
            beta_str, gamma_str, ss_dir, "\n  ".join(candidates)
        )
    )
    return None


def spawn_tailornet_garment_smpl(
    garment_class,
    gender,
    beta_str,
    gamma_str,
    clearScene=False,
):
    """
    Spawn a TailorNet garment in Maya *the same way visualize_dataset.py does*:
    - Load beta, apose and unposed garment displacements
    - Run SMPLNP to get (body_v, gar_v)
    - Use gar_v + garment faces to build the mesh in Maya

    Args:
        garment_class: e.g. "t-shirt", "shirt", "pant", "skirt", "short-pant", "old-t-shirt"
        gender: "female" or "male" (TailorNet uses <garment_class>_<gender> dirs)
        beta_str: shape index as string, e.g. "000"
        gamma_str: style index as string, e.g. "023"
        clearScene: if True, deletes all transforms except default cameras
    """

    if clearScene:
        _clear_scene_keep_cameras()

    # 1) Paths
    class_dir = osp.join(TN_DATASET, f"{garment_class}_{gender}")
    if not osp.exists(class_dir):
        cmds.error("TailorNet class dir not found: {}".format(class_dir))
        return

    shape_dir = osp.join(class_dir, "shape")
    if not osp.exists(shape_dir):
        cmds.error("shape/ dir not found: {}".format(shape_dir))
        return

    apose_path = osp.join(TN_DATASET, "dataset_meta", "apose.npy")
    if not osp.exists(apose_path):
        cmds.error("apose.npy not found at: {}".format(apose_path))
        return

    # 2) Resolve file names as in visualize_dataset.py
    #    beta file: 'beta_{}.npy' OR 'beta_ {}.npy'
    beta_candidates = [
        osp.join(shape_dir, f"beta_{beta_str}.npy"),
        osp.join(shape_dir, f"beta_ {beta_str}.npy"),
    ]
    beta_path = None
    for p in beta_candidates:
        if osp.exists(p):
            beta_path = p
            break
    if beta_path is None:
        cmds.error(
            "Could not find beta file for {}: tried\n  {}".format(
                beta_str, "\n  ".join(beta_candidates)
            )
        )
        return

    ss_path = _find_style_shape_file(class_dir, beta_str, gamma_str)
    if ss_path is None:
        return

    # 3) Load data
    apose = np.load(apose_path)         # (72,) or (1,72) pose in A-pose
    beta = np.load(beta_path)           # (10,) shape
    unpose_v = np.load(ss_path)         # (Nv, 3) garment offsets (unposed)

    if unpose_v.ndim != 2 or unpose_v.shape[1] != 3:
        cmds.error("Unexpected unpose_v shape in {}: {}".format(ss_path, unpose_v.shape))
        return

    # 4) Use TailorNet's SMPLNP to get *final* body_v, gar_v
    #    (this is the step my initial script was missing)
    smpl = SMPLNP(gender=gender, cuda=False)
    body_v, gar_v = smpl(beta, apose, unpose_v, garment_class, batch=False)
    # gar_v: (Nv, 3) garment vertices in TailorNet "canonical" pose

    # 5) Get faces from garment_class_info.pkl
    faces = _load_garment_faces(garment_class)
    if faces is None:
        return

    # 6) (Optional) Rotate/center like visualize_garment_body if you want
    # For now we just drop them into Maya in TailorNet coordinates.
    mesh_name = f"{garment_class}_{gender}_b{beta_str}_g{gamma_str}"
    created = _create_mesh_in_maya(gar_v, faces, name=mesh_name)
    print("Spawned TailorNet garment (via SMPLNP):", created)


try:
    spawn_tailornet_garment_smpl(
        garment_class="t-shirt",
        gender="male",
        beta_str="000",
        gamma_str="000",
        clearScene=True,
    )
except Exception as e:
    print(traceback.format_exc())