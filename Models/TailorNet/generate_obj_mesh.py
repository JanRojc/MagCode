import os
import os.path as osp
import sys
import pickle
import numpy as np
import importlib.util

# ============================================================
# CONFIG – EDIT THESE
# ============================================================

# Path to your ContourCraft repo (directory that contains "utils/")
CONTOURCRAFT_REPO = r"C:\Users\janro\Documents\Faks\MAG\MagCode\Models\ContourCraft"

# TailorNet garment to convert
GARMENT_CLASS = "shirt"   # e.g. "t-shirt", "shirt", "pant", "skirt", "short-pant", "old-t-shirt"
GENDER        = "male"    # "male" or "female"

# One specific TailorNet style-shape pair (from <garment_class>_<gender>/avail.txt)
BETA_STR      = "000"     # e.g. "000"
GAMMA_STR     = "000"     # e.g. "023"

# GarmentCreator sampling (0 is fine for fitted tops/pants)
N_LBS_SAMPLES = 0         # use >0 (e.g. 1000) for very loose garments

# ============================================================
# Helper: load a module from an explicit file path
# ============================================================

def _load_module(name: str, path: str):
    """Load a module from file and register it in sys.modules under the given name."""
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not create spec for {name} from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod

# ============================================================
# Load ContourCraft's utils.* modules explicitly
# ============================================================

# Remove any previously loaded "utils.*" (e.g. from TailorNet)
for key in list(sys.modules.keys()):
    if key == "utils" or key.startswith("utils."):
        sys.modules.pop(key, None)

utils_root = osp.join(CONTOURCRAFT_REPO, "utils")
if not osp.isdir(utils_root):
    raise RuntimeError(f"'utils' directory not found at {utils_root}")

# Load ContourCraft's utils modules directly
defaults_mod = _load_module("utils.defaults",    osp.join(utils_root, "defaults.py"))
io_mod       = _load_module("utils.io",          osp.join(utils_root, "io.py"))
mesh_mod     = _load_module("utils.mesh_creation", osp.join(utils_root, "mesh_creation.py"))

from utils.defaults import DEFAULTS
from utils.io import save_obj
from utils.mesh_creation import GarmentCreator

# ============================================================
# Now import TailorNet code
# ============================================================

THIS_DIR = osp.dirname(osp.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.append(THIS_DIR)

from global_var import ROOT as TN_ROOT         # TailorNet dataset ROOT
from smpl_torch import SMPLNP                  # TailorNet SMPL wrapper


# ============================================================
# TailorNet helpers
# ============================================================

def _find_beta_path(shape_dir: str, beta_str: str) -> str:
    """Try a few common naming patterns for TailorNet beta files."""
    candidates = [
        osp.join(shape_dir, f"beta_{beta_str}.npy"),
        osp.join(shape_dir, f"beta_ {beta_str}.npy"),
    ]
    for p in candidates:
        if osp.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find beta file for {beta_str} in {shape_dir}.\nTried:\n  " +
        "\n  ".join(candidates)
    )


def _find_style_shape_path(ss_dir: str, beta_str: str, gamma_str: str) -> str:
    """Try a few patterns for TailorNet style_shape filenames."""
    candidates = [
        osp.join(ss_dir, f"beta{beta_str}_gamma{gamma_str}.npy"),
        osp.join(ss_dir, f"beta_{beta_str}_gamma_{gamma_str}.npy"),
        osp.join(ss_dir, f"beta {beta_str}_gamma {gamma_str}.npy"),
    ]
    for p in candidates:
        if osp.exists(p):
            return p
    raise FileNotFoundError(
        f"Could not find style_shape file for beta={beta_str}, gamma={gamma_str} in {ss_dir}.\n"
        f"Tried:\n  " + "\n  ".join(candidates)
    )


def get_tailornet_garment_mesh(
    garment_class: str,
    gender: str,
    beta_str: str,
    gamma_str: str,
):
    """
    Reconstruct a TailorNet garment in canonical pose (A-pose) using SMPLNP.

    Returns:
        gar_v: (Nv, 3) garment vertices
        faces: (F, 3) int triangle indices
    """
    class_dir = osp.join(TN_ROOT, f"{garment_class}_{gender}")
    if not osp.exists(class_dir):
        raise FileNotFoundError(f"TailorNet class dir not found: {class_dir}")

    shape_dir = osp.join(class_dir, "shape")
    ss_dir    = osp.join(class_dir, "style_shape")

    apose_path = osp.join(TN_ROOT, "apose.npy")
    if not osp.exists(apose_path):
        raise FileNotFoundError(f"apose.npy not found at {apose_path}")

    beta_path = _find_beta_path(shape_dir, beta_str)
    ss_path   = _find_style_shape_path(ss_dir, beta_str, gamma_str)

    apose   = np.load(apose_path)
    beta    = np.load(beta_path)
    unpose_v = np.load(ss_path)

    if unpose_v.ndim != 2 or unpose_v.shape[1] != 3:
        raise RuntimeError(f"Unexpected unpose_v shape in {ss_path}: {unpose_v.shape}")

    smpl = SMPLNP(gender=gender, cuda=False)
    body_v, gar_v = smpl(beta, apose, unpose_v, garment_class, batch=False)

    # Faces from garment_class_info.pkl
    meta_path = osp.join(TN_ROOT, "garment_class_info.pkl")
    if not osp.exists(meta_path):
        raise FileNotFoundError(f"garment_class_info.pkl not found at {meta_path}")

    with open(meta_path, "rb") as f:
        garment_meta = pickle.load(f, encoding="latin1")

    if garment_class not in garment_meta:
        raise KeyError(f"Garment class '{garment_class}' not found in garment_class_info.pkl")

    faces = np.asarray(garment_meta[garment_class]["f"], dtype=np.int32)
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise RuntimeError(f"Unexpected faces shape for {garment_class}: {faces.shape}")

    return gar_v, faces


# ============================================================
# TailorNet -> ContourCraft conversion
# ============================================================

def convert_one_to_contourcraft():
    """
    TailorNet → ContourCraft for a single (BETA_STR, GAMMA_STR):
      - reconstruct TailorNet garment in canonical pose
      - save OBJ in aux_data/garment_meshes/smpl
      - run GarmentCreator to create garment_dict .pkl
    """

    gar_v, faces = get_tailornet_garment_mesh(
        GARMENT_CLASS, GENDER, BETA_STR, GAMMA_STR
    )

    garment_name = f"{GARMENT_CLASS}_{GENDER}_b{BETA_STR}_g{GAMMA_STR}"

    aux_root = DEFAULTS.aux_data

    # 1) Save OBJ
    mesh_dir = osp.join(aux_root, "garment_meshes", "smpl")
    os.makedirs(mesh_dir, exist_ok=True)

    obj_path = osp.join(mesh_dir, garment_name + ".obj")
    save_obj(obj_path, gar_v, faces)
    print("[TailorNet -> OBJ]", obj_path)

    # 2) Create garment_dict via GarmentCreator
    body_models_root  = osp.join(aux_root, "body_models")
    garment_dicts_dir = osp.join(aux_root, "garment_dicts", "smpl")
    os.makedirs(garment_dicts_dir, exist_ok=True)

    gc = GarmentCreator(
        garment_dicts_dir,
        body_models_root,
        "smpl",
        GENDER,
        n_samples_lbs=N_LBS_SAMPLES,
        coarse=True,
        approximate_center=True,
    )

    gc.add_garment(obj_path, garment_name)

    pkl_path = osp.join(garment_dicts_dir, garment_name + ".pkl")
    print("[GarmentCreator] Created garment dict:", pkl_path)


def main():
    print("CONTOURCRAFT_REPO =", CONTOURCRAFT_REPO)
    print("TN_ROOT =", TN_ROOT)
    convert_one_to_contourcraft()


if __name__ == "__main__":
    main()
