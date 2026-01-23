import os
import os.path as osp
import pickle
import numpy as np

from global_var import ROOT as TN_ROOT
from smpl_torch import SMPLNP  # works in your TailorNet env

# ---- config ----
GARMENT_CLASS = "t-shirt"   # e.g. "t-shirt", "shirt", "pant", ...
GENDER        = "female"
BETA_STR      = "000"     # from shirt_male/avail.txt
GAMMA_STR     = "000"     # from shirt_male/avail.txt 

def main():
    class_dir = osp.join(TN_ROOT, f"{GARMENT_CLASS}_{GENDER}")
    shape_dir = osp.join(class_dir, "shape")
    ss_dir    = osp.join(class_dir, "style_shape")
    out_pkl = f"{class_dir}/tailornet_verts_faces_tmp.pkl"

    apose_path = osp.join(TN_ROOT, "dataset_meta", "apose.npy")
    beta_path  = osp.join(shape_dir, f"beta_{BETA_STR}.npy")
    ss_path    = osp.join(ss_dir,  f"beta{BETA_STR}_gamma{GAMMA_STR}.npy")

    apose   = np.load(apose_path)
    beta    = np.load(beta_path)
    unpose_v = np.load(ss_path)

    smpl = SMPLNP(gender=GENDER, cuda=False)
    body_v, gar_v = smpl(beta, apose, unpose_v, GARMENT_CLASS, batch=False)

    meta_path = osp.join(TN_ROOT, "dataset_meta", "garment_class_info.pkl")
    with open(meta_path, "rb") as f:
        garment_meta = pickle.load(f, encoding="latin1")
    faces = np.asarray(garment_meta[GARMENT_CLASS]["f"], dtype=np.int32)

    data = {
        "verts": gar_v.astype(np.float32),
        "faces": faces.astype(np.int32),
        "garment_class": GARMENT_CLASS,
        "gender": GENDER,
        "beta_str": BETA_STR,
        "gamma_str": GAMMA_STR,
    }

    os.makedirs(osp.dirname(out_pkl), exist_ok=True)
    with open(out_pkl, "wb") as f:
        pickle.dump(data, f)

    print("Wrote PKL:", out_pkl)
    print("verts:", data["verts"].shape, "faces:", data["faces"].shape)

if __name__ == "__main__":
    main()
