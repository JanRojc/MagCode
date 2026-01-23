import os
import os.path as osp
import sys
import pickle
import numpy as np

GARMENT_CLASS = "t-shirt"   # e.g. "t-shirt", "shirt", "pant", ...
GENDER        = "female"

TN_DATA = "/Users/jan.rojc/Documents/MagCode/Data/TailorNet"
CONTOURCRAFT_REPO = "/home/janr/documents/MagCode/Models/ContourCraft"

MODEL_TYPE = "smpl"   # ContourCraft model type
N_LBS_SAMPLES = 0     # GarmentCreator sampling

if CONTOURCRAFT_REPO not in sys.path:
    sys.path.append(CONTOURCRAFT_REPO)

from utils.defaults import DEFAULTS
from utils.io import save_obj
from utils.mesh_creation import GarmentCreator


def add_auto_pins(garment_pkl_path: str, top_ratio: float = 0.1):
    """
    Mark vertices in the top `top_ratio` of their Y-extent as pinned.

    Assumes a HOOD/ContourCraft-style garment dict with:
      - 'rest_pos': (N, 3) float array of template vertices
      - optional 'node_type': (N,) int array, 0=regular, 3=pinned

    Writes the modified dict back to `garment_pkl_path`.
    """
    with open(garment_pkl_path, "rb") as f:
        gdict = pickle.load(f)

    if "rest_pos" not in gdict:
        raise KeyError(
            f"'rest_pos' not found in garment dict {garment_pkl_path}. "
            "Check the keys in this file to match your ContourCraft version."
        )

    rest_pos = np.asarray(gdict["rest_pos"])
    if rest_pos.ndim != 2 or rest_pos.shape[1] != 3:
        raise ValueError(
            f"Unexpected rest_pos shape {rest_pos.shape} in {garment_pkl_path}"
        )

    y = rest_pos[:, 1]
    y_min, y_max = float(y.min()), float(y.max())
    if y_max <= y_min:
        # Degenerate (flat) Y-range – nothing sensible to pin
        print(f"[auto pins] Skipping pins for {garment_pkl_path}: y_max == y_min")
        return []

    # Threshold for "top X% in Y"
    thresh = y_min + (y_max - y_min) * (1.0 - top_ratio)
    pinned_mask = y >= thresh
    pinned_indices = np.nonzero(pinned_mask)[0]

    if pinned_indices.size == 0:
        print(f"[auto pins] No vertices in top {top_ratio*100:.1f}% for {garment_pkl_path}")
        return []

    # node_type: 0 = regular, 3 = pinned (HOOD/ContourCraft convention)
    node_type = gdict.get("node_type", None)
    if node_type is None:
        node_type = np.zeros(rest_pos.shape[0], dtype=np.int64)
    else:
        node_type = np.asarray(node_type).copy()
        if node_type.shape[0] != rest_pos.shape[0]:
            raise ValueError(
                f"node_type length {node_type.shape[0]} != N verts {rest_pos.shape[0]} "
                f"in {garment_pkl_path}"
            )

    node_type[pinned_indices] = 3
    gdict["node_type"] = node_type

    with open(garment_pkl_path, "wb") as f:
        pickle.dump(gdict, f)

    print(
        f"[auto pins] Pinned {pinned_indices.size} verts "
        f"(top {top_ratio*100:.1f}% in Y) in {garment_pkl_path}"
    )
    return pinned_indices.tolist()


def main():
    class_dir = osp.join(TN_DATA, f"{GARMENT_CLASS}_{GENDER}")
    in_pkl = f"{class_dir}/tailornet_verts_faces_tmp.pkl"
    with open(in_pkl, "rb") as f:
        data = pickle.load(f)

    verts = data["verts"]
    faces = data["faces"]
    gender = data.get("gender", "male")
    beta_str = data.get("beta_str", "000")
    gamma_str = data.get("gamma_str", "000")

    garment_obj_name = f"ccraft_{beta_str}_{gamma_str}"
    garment_dict_name = f"ccraft_{GARMENT_CLASS}_{GENDER}_{beta_str}_{gamma_str}"

    aux_root = DEFAULTS.aux_data

    # 1) write OBJ using ContourCraft's save_obj
    mesh_dir = osp.join(aux_root, "garment_meshes", MODEL_TYPE)
    os.makedirs(mesh_dir, exist_ok=True)
    obj_path = osp.join(class_dir, garment_obj_name + ".obj")
    save_obj(obj_path, verts, faces)
    print("[PKL -> OBJ]", obj_path)

    # 2) run GarmentCreator
    body_models_root  = osp.join(aux_root, "body_models")
    garment_dicts_dir = osp.join(aux_root, "garment_dicts", MODEL_TYPE)
    os.makedirs(garment_dicts_dir, exist_ok=True)

    gc = GarmentCreator(
        garment_dicts_dir,
        body_models_root,
        MODEL_TYPE,
        gender,
        n_samples_lbs=N_LBS_SAMPLES,
        coarse=True,
        approximate_center=True,
    )

    gc.add_garment(obj_path, garment_dict_name)

    # Path of the newly created garment dict
    pkl_path = osp.join(garment_dicts_dir, garment_dict_name + ".pkl")
    print("[GarmentCreator] Created garment dict:", pkl_path)

    # 3) Automatically mark top 10% Y verts as pinned
    pinned = add_auto_pins(pkl_path, top_ratio=0.10)
    print("[auto pins] Example indices:", pinned[:20])

    # 4) Clean up intermediate TailorNet pkl
    os.remove(in_pkl)
    print("[cleanup] Removed intermediate:", in_pkl)


if __name__ == "__main__":
    main()
