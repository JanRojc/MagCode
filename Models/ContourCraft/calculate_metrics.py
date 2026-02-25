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
# MATH ENGINE (The 'Driver and Passenger' Logic)
# ======================================================================================
def get_rotation_matrix(axis, angle_deg):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 'x': return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'z': return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def solve_icp_transform_fast(src, dst, iters=30, sample_size=1000):
    c_src_init = np.mean(src, axis=0)
    c_dst_init = np.mean(dst, axis=0)
    curr_src_full = src - c_src_init
    target_centered = dst - c_dst_init
    total_R = np.eye(3)
    tree = cKDTree(target_centered)

    for _ in range(iters):
        idx = np.random.choice(len(curr_src_full), min(sample_size, len(curr_src_full)), replace=False)
        src_sample = curr_src_full[idx]
        _, indices_dst = tree.query(src_sample)
        matched_dst = target_centered[indices_dst]
        S = src_sample.T @ matched_dst
        U, _, Vt = np.linalg.svd(S)
        R = Vt.T @ U.T
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = Vt.T @ U.T
        curr_src_full = curr_src_full @ R.T
        total_R = R @ total_R

    final_t = c_dst_init - (total_R @ c_src_init)
    return total_R, final_t

# ======================================================================================
# DATA LOADERS
# ======================================================================================
def load_mesh_data(path):
    if not os.path.exists(path): return None
    m = trimesh.load(path)
    return np.array(m.vertices, dtype=np.float32)

def load_pkl_frame(path, target_type, frame):
    if not os.path.exists(path): return None
    with open(path, 'rb') as f:
        data = pickle.load(f)
    v_keys = {'cloth': ['pred', 'pred_pos', 'vertices'], 'body': ['obstacle', 'body', 'smpl_verts']}
    v = None
    for k in v_keys[target_type]:
        if k in data:
            v = data[k]
            break
    if v is None: return None
    if len(v.shape) == 3: v = v[max(0, min(frame, v.shape[0]-1))]
    if hasattr(v, 'detach'): v = v.detach().cpu().numpy()
    return np.array(v, dtype=np.float32)

# ======================================================================================
# CORE EVALUATION FUNCTION
# ======================================================================================
def evaluate_experiment(seq_num, seq_idx, gender, garment, garment_type):
    seq_args = [seq_num, seq_idx, gender, garment, garment_type]
    R_tailor_fix = get_rotation_matrix('z', -90) @ get_rotation_matrix('x', -90)
    
    # Store frame-by-frame errors to average them later
    model_errors = {m: [] for m in MODELS.keys()}
    
    start_frame, end_frame = 5, 300

    for frame in range(start_frame, end_frame):
        # 1. Load Maya GT
        gt_base = os.path.join(RESULTS_ROOT, GT_MODEL_FOLDER, *seq_args, "result_ply_files")
        v_body_gt = load_mesh_data(os.path.join(gt_base, f"body_{frame:04d}.ply"))
        v_cloth_gt = load_mesh_data(os.path.join(gt_base, f"pred_gar_{frame:04d}.ply"))
        if v_body_gt is None or v_cloth_gt is None: continue

        for model_name, folder in MODELS.items():
            model_base = os.path.join(RESULTS_ROOT, folder, *seq_args)
            curr_frame = frame - 2 if model_name == "HOOD" else frame
            
            # 2. Load Predictions
            if model_name in ["CCraft", "HOOD"]:
                pkl_path = os.path.join(model_base, "output.pkl")
                v_body_p = load_pkl_frame(pkl_path, "body", curr_frame)
                v_cloth_p = load_pkl_frame(pkl_path, "cloth", curr_frame)
            else: # TailorNet
                ply_base = os.path.join(model_base, "result_ply_files")
                v_body_p = load_mesh_data(os.path.join(ply_base, f"body_{curr_frame:04d}.ply"))
                v_cloth_p = load_mesh_data(os.path.join(ply_base, f"pred_gar_{curr_frame:04d}.ply"))

            if v_body_p is None or v_cloth_p is None: continue

            # 3. Align & Transform
            if model_name == "TailorNet":
                v_body_p = v_body_p @ R_tailor_fix.T
                v_cloth_p = v_cloth_p @ R_tailor_fix.T

            R, t = solve_icp_transform_fast(v_body_p, v_body_gt)
            v_cloth_aligned = (v_cloth_p @ R.T) + t

            # 4. RMSE Calculation
            if len(v_cloth_aligned) == len(v_cloth_gt):
                rmse = np.sqrt(np.mean(np.linalg.norm(v_cloth_aligned - v_cloth_gt, axis=1)**2))
                model_errors[model_name].append(rmse)

    # Return averages for the whole sequence
    row = {
        "sequence": f"{seq_num}_{seq_idx}",
        "gender": gender,
        "garment": garment,
        "material": garment_type
    }
    for m in MODELS.keys():
        row[f"{m}_avg_rmse"] = np.mean(model_errors[m]) if model_errors[m] else None
        
    return row

# ======================================================================================
# MAIN BATCH LOOP
# ======================================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)
    
    sequences = ["01", "02", "05", "07"]
    sequence_indices = ["01", "02", "03", "04", "05"]
    gender_list = ["male"]
    garments = ["t-shirt", "shirt", "pant"]
    garment_types = ["cotton", "silk"]

    summary_data = []

    for seq_num in sequences:
        for seq_idx in sequence_indices:
            for gender in gender_list:
                for garment in garments:
                    for garment_type in garment_types:
                        # Logic check from user
                        if (garment, gender) == ("shirt", "female"): 
                            continue
                        
                        print(f"\n--- Evaluating: {seq_num}_{seq_idx} | {gender} | {garment} | {garment_type} ---")
                        
                        try:
                            result_row = evaluate_experiment(seq_num, seq_idx, gender, garment, garment_type)
                            summary_data.append(result_row)
                        except Exception as e:
                            print(f"Error in experiment: {e}")

    # Save final flattened CSV
    df = pd.DataFrame(summary_data)
    df.to_csv(os.path.join(OUTPUT_ROOT, "comprehensive_cloth_metrics.csv"), index=False)
    print(f"\nSaved summary of {len(summary_data)} experiments to {OUTPUT_ROOT}")