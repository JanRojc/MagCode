import os
import pickle
import numpy as np
import trimesh
import pandas as pd
from scipy.spatial import cKDTree

# ======================================================================================
# CONFIGURATION
# ======================================================================================
RESULTS_ROOT = "/mnt/d/ClothSim/Results"
OUTPUT_ROOT = os.path.join(RESULTS_ROOT, "CalculatedMetrics")
MODELS = {"TailorNet": "TailorNet", "CCraft": "ccraft", "HOOD": "hood"}
GT_MODEL_FOLDER = "Maya"

# ======================================================================================
# MATH HELPERS
# ======================================================================================
def get_rotation_matrix(axis, angle_deg):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 'x': return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'z': return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def solve_icp_transform_fast(src, dst, iters=30, sample_size=1000):
    """
    Enhanced ICP with better initialization to prevent the 'slight tilt'.
    """
    # 1. INITIAL CENTROID ALIGNMENT (Mimics Maya's pivot logic)
    c_src_init = np.mean(src, axis=0)
    c_dst_init = np.mean(dst, axis=0)
    
    curr_src_full = src - c_src_init
    target_centered = dst - c_dst_init
    
    total_R = np.eye(3)
    tree = cKDTree(target_centered)

    for _ in range(iters):
        # Random sampling for speed
        idx = np.random.choice(len(curr_src_full), min(sample_size, len(curr_src_full)), replace=False)
        src_sample = curr_src_full[idx]
        
        _, indices_dst = tree.query(src_sample)
        matched_dst = target_centered[indices_dst]

        # Kabsch Algorithm (Rotation Only on centered data)
        S = src_sample.T @ matched_dst
        U, _, Vt = np.linalg.svd(S)
        R = Vt.T @ U.T
        
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
            
        # Update
        curr_src_full = curr_src_full @ R.T
        total_R = R @ total_R

    # 2. FINAL TRANSLATION (Map the rotated centroid back to the target's world space)
    final_t = c_dst_init - (total_R @ c_src_init)
    
    return total_R, final_t

# ======================================================================================
# DATA LOADERS
# ======================================================================================
def load_mesh_data(path):
    if not os.path.exists(path): return None, None
    m = trimesh.load(path)
    return np.array(m.vertices, dtype=np.float32), m.faces

def load_pkl_frame_with_faces(path, target_type, frame):
    if not os.path.exists(path): return None, None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    
    v_keys = {'cloth': ['pred', 'pred_pos', 'vertices'], 'body': ['obstacle', 'body', 'smpl_verts']}
    f_keys = {'cloth': ['cloth_faces', 'faces'], 'body': ['obstacle_faces', 'body_faces', 'faces']}
    
    v, f_idx = None, None
    for k in v_keys[target_type]:
        if k in data:
            v = data[k]
            break
    for k in f_keys[target_type]:
        if k in data:
            f_idx = data[k]
            break

    if v is None: return None, None
    if len(v.shape) == 3:
        v = v[max(0, min(frame, v.shape[0]-1))]
    
    if hasattr(v, 'detach'): v = v.detach().cpu().numpy()
    if hasattr(f_idx, 'detach'): f_idx = f_idx.detach().cpu().numpy()
    return np.array(v, dtype=np.float32), f_idx

# ======================================================================================
# RUN EVALUATION
# ======================================================================================
def run_evaluation(seq_num, seq_idx, gender, garment, cloth_type):
    seq_args = [seq_num, seq_idx, gender, garment, cloth_type]
    debug_dir = os.path.join(OUTPUT_ROOT, "debug_meshes")
    os.makedirs(debug_dir, exist_ok=True)

    # TAILORNET PRE-ROTATION (Applied to raw data)
    R_tailor_fix = get_rotation_matrix('z', -90) @ get_rotation_matrix('x', -90)

    all_results = []
    start_frame = 5
    end_frame = 10 

    for frame in range(start_frame, end_frame):
        print(f"Processing Frame {frame}/{end_frame}...", end="\r")
        
        # 1. LOAD MAYA GT
        gt_base = os.path.join(RESULTS_ROOT, GT_MODEL_FOLDER, *seq_args, "result_ply_files")
        v_body_gt, _ = load_mesh_data(os.path.join(gt_base, f"body_{frame:04d}.ply"))
        v_cloth_gt, f_cloth_gt = load_mesh_data(os.path.join(gt_base, f"pred_gar_{frame:04d}.ply"))
        if v_body_gt is None: continue

        for model_name, folder in MODELS.items():
            model_base = os.path.join(RESULTS_ROOT, folder, *seq_args)
            curr_frame = frame - 2 if model_name == "HOOD" else frame
            
            # 2. LOAD PREDICTIONS
            if model_name in ["CCraft", "HOOD"]:
                v_body_p, _ = load_pkl_frame_with_faces(os.path.join(model_base, "output.pkl"), "body", curr_frame)
                v_cloth_p, f_cloth_p = load_pkl_frame_with_faces(os.path.join(model_base, "output.pkl"), "cloth", curr_frame)
            else:
                v_body_p, _ = load_mesh_data(os.path.join(model_base, "result_ply_files", f"body_{curr_frame:04d}.ply"))
                v_cloth_p, f_cloth_p = load_mesh_data(os.path.join(model_base, "result_ply_files", f"pred_gar_{curr_frame:04d}.ply"))

            if v_body_p is None or v_cloth_p is None: continue

            # 3. TAILORNET INITIAL FIX
            if model_name == "TailorNet":
                v_body_p = v_body_p @ R_tailor_fix.T
                v_cloth_p = v_cloth_p @ R_tailor_fix.T

            # 4. ALIGNMENT SOLVER (Body to Body)
            R, t = solve_icp_transform_fast(v_body_p, v_body_gt)
            
            # Apply to cloth
            v_cloth_aligned = (v_cloth_p @ R.T) + t

            # 5. METRICS
            if len(v_cloth_aligned) == len(v_cloth_gt):
                dist = np.linalg.norm(v_cloth_aligned - v_cloth_gt, axis=1)
                rmse = np.sqrt(np.mean(dist**2))
                max_e = np.max(dist)
                all_results.append({"frame": frame, "model": model_name, "rmse": rmse, "max_error": max_e})

            # 6. EXPORT
            if frame == start_frame:
                trimesh.Trimesh(v_cloth_gt, f_cloth_gt, process=False).export(os.path.join(debug_dir, "ref_maya.obj"))
                trimesh.Trimesh(v_cloth_aligned, f_cloth_p, process=False).export(os.path.join(debug_dir, f"aligned_{model_name}.obj"))

    return pd.DataFrame(all_results)

if __name__ == "__main__":
    df = run_evaluation("01", "01", "male", "t-shirt", "cotton")
    df.to_csv(os.path.join(OUTPUT_ROOT, "final_stable_eval.csv"), index=False)