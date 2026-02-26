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
# HIGH-SPEED METRIC HELPERS
# ======================================================================================
def calculate_collision_rate_fast(cloth_verts, body_verts, body_normals):
    """
    Ultra-fast collision check using KDTree and surface normals.
    Dot product < 0 means the cloth vertex is 'behind' the body surface.
    """
    tree = cKDTree(body_verts)
    _, idx = tree.query(cloth_verts)
    vec = cloth_verts - body_verts[idx]
    # Negative dot product means inside
    inside = np.sum(vec * body_normals[idx], axis=1) < 0
    return (np.sum(inside) / len(cloth_verts)) * 100

def compute_acceleration(v_prev, v_curr, v_next):
    """a_t = v_{t+1} - 2v_t + v_{t-1}"""
    return v_next - 2 * v_curr + v_prev

# ======================================================================================
# DATA LOADERS
# ======================================================================================
def get_rotation_matrix(axis, angle_deg):
    angle_rad = np.radians(angle_deg)
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    if axis == 'x': return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == 'z': return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

def solve_icp_transform_fast(src, dst, iters=30, sample_size=1000):
    c_src_init, c_dst_init = np.mean(src, axis=0), np.mean(dst, axis=0)
    curr_src_full, target_centered = src - c_src_init, dst - c_dst_init
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

def load_mesh_all(path):
    """Returns (vertices, faces, normals) for Maya GT meshes."""
    if not os.path.exists(path): return None, None, None
    m = trimesh.load(path, process=False)
    return np.array(m.vertices), m.faces, np.array(m.vertex_normals)

def load_mesh_verts_only(path):
    """Returns just vertices for TailorNet/Maya PLY files."""
    if not os.path.exists(path): return None
    m = trimesh.load(path, process=False)
    return np.array(m.vertices)

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

def save_debug_meshes(model_name, v_cloth_gt, f_cloth_gt, v_aligned, v_body_gt, f_body_gt, v_body_p):
    debug_dir = os.path.join(OUTPUT_ROOT, "debug_meshes")
    os.makedirs(debug_dir, exist_ok=True)
    
    # 1. Save Maya GT (The anchor)
    trimesh.Trimesh(vertices=v_cloth_gt, faces=f_cloth_gt, process=False).export(
        os.path.join(debug_dir, "garment_maya.obj")
    )
    # 2. Save Aligned Prediction
    # Using f_cloth_p ensures Maya treats it as a surface, not a point cloud
    trimesh.Trimesh(vertices=v_aligned, faces=f_cloth_gt, process=False).export(
        os.path.join(debug_dir, f"garment_aligned_{model_name}.obj")
    )
    # 3. Save Maya Body (Optional: to verify body-to-body overlap)
    trimesh.Trimesh(vertices=v_body_gt, faces=f_body_gt, process=False).export(
        os.path.join(debug_dir, "body_maya.obj")
    )
    # 4. Save Model Body
    trimesh.Trimesh(vertices=v_body_p, faces=f_body_gt, process=False).export(
        os.path.join(debug_dir, f"body_{model_name}.obj")
    )


# ======================================================================================
# CORE EVALUATION
# ======================================================================================
def evaluate_experiment(seq_num, seq_idx, gender, garment, garment_type):
    seq_args = [seq_num, seq_idx, gender, garment, garment_type]
    z_deg = -90
    x_deg = -90
    if seq_num =="01" and seq_idx != "01" and seq_idx != "04":
        x_deg = -90
        z_deg = 90
    R_tailor_fix = get_rotation_matrix('z', z_deg) @ get_rotation_matrix('x', x_deg)
    
    metrics = {m: {"rmse": [], "acc_err": [], "col_rate": []} for m in MODELS.keys()}
    history_gt = []
    history_pred = {m: [] for m in MODELS.keys()}

    start_frame, end_frame = 5, 300

    for frame in range(start_frame, end_frame):
        print(f"Calculating frame {frame}/{end_frame}", end="\r")

        # 1. LOAD MAYA GT
        gt_base = os.path.join(RESULTS_ROOT, GT_MODEL_FOLDER, *seq_args, "result_ply_files")
        v_body_gt, f_body_gt, n_body_gt = load_mesh_all(os.path.join(gt_base, f"body_{frame:04d}.ply"))
        v_cloth_gt, f_cloth_gt, _ = load_mesh_all(os.path.join(gt_base, f"pred_gar_{frame:04d}.ply"))
        
        if v_body_gt is None or v_cloth_gt is None: continue
        history_gt.append(v_cloth_gt)
        if len(history_gt) > 3: history_gt.pop(0)

        for model_name, folder in MODELS.items():
            model_base = os.path.join(RESULTS_ROOT, folder, *seq_args)
            curr_frame = frame - 2 if model_name == "HOOD" else frame
            
            # 2. LOAD (Restored Logic)
            if model_name in ["CCraft", "HOOD"]:
                pkl_path = os.path.join(model_base, "output.pkl")
                v_body_p = load_pkl_frame(pkl_path, "body", curr_frame)
                v_cloth_p = load_pkl_frame(pkl_path, "cloth", curr_frame)
            else:
                ply_base = os.path.join(model_base, "result_ply_files")
                v_body_p = load_mesh_verts_only(os.path.join(ply_base, f"body_{curr_frame:04d}.ply"))
                v_cloth_p = load_mesh_verts_only(os.path.join(ply_base, f"pred_gar_{curr_frame:04d}.ply"))

            if v_body_p is None or v_cloth_p is None: continue
            
            # 3. ALIGN & TRANSFORM
            if model_name == "TailorNet":
                v_body_p, v_cloth_p = v_body_p @ R_tailor_fix.T, v_cloth_p @ R_tailor_fix.T

            R, t = solve_icp_transform_fast(v_body_p, v_body_gt)
            v_aligned = (v_cloth_p @ R.T) + t

            # --- DEBUG EXPORT  ---
            if SAVE_DEBUG_MESHES and frame == save_mesh_frame:
                v_aligned_body = (v_body_p @ R.T) + t
                save_debug_meshes(model_name, v_cloth_gt, f_cloth_gt, v_aligned, v_body_gt, f_body_gt, v_aligned_body)

            # --- METRICS ---
            # RMSE
            if len(v_aligned) == len(v_cloth_gt):
                metrics[model_name]["rmse"].append(np.sqrt(np.mean(np.linalg.norm(v_aligned - v_cloth_gt, axis=1)**2)))

            # COLLISION (Fast Normal Check)
            metrics[model_name]["col_rate"].append(calculate_collision_rate_fast(v_aligned, v_body_gt, n_body_gt))

            # ACCELERATION ERROR
            history_pred[model_name].append(v_aligned)
            if len(history_pred[model_name]) == 3 and len(history_gt) == 3:
                a_pred = compute_acceleration(history_pred[model_name][0], history_pred[model_name][1], history_pred[model_name][2])
                a_gt = compute_acceleration(history_gt[0], history_gt[1], history_gt[2])
                # Difference between prediction acceleration and GT acceleration
                metrics[model_name]["acc_err"].append(np.mean(np.linalg.norm(a_pred - a_gt, axis=1)))
                history_pred[model_name].pop(0)


    # SUMMARIZE
    row = {"sequence": f"{seq_num}_{seq_idx}", "gender": gender, "garment": garment, "material": garment_type}
    for m in MODELS.keys():
        row[f"{m}_rmse"] = np.mean(metrics[m]["rmse"]) if metrics[m]["rmse"] else None
        row[f"{m}_accel_err"] = np.mean(metrics[m]["acc_err"]) if metrics[m]["acc_err"] else None
        row[f"{m}_collision"] = np.mean(metrics[m]["col_rate"]) if metrics[m]["col_rate"] else None
    return row


# ======================================================================================
# MAIN BATCH LOOP
# ======================================================================================
if __name__ == "__main__":
    os.makedirs(OUTPUT_ROOT, exist_ok=True)

    SAVE_DEBUG_MESHES = True
    save_mesh_frame = 5
    
    sequences = ["01", "02", "05", "07"]
    sequence_indices = ["01", "02", "03", "04", "05"]
    gender_list = ["male"]
    garments = ["t-shirt", "shirt", "pant"]
    garment_types = ["cotton", "silk"]

    sequences = ["01"]
    sequence_indices = ["01"]
    gender_list = ["male"]
    garments = ["pant"]
    garment_types = ["cotton"]

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
    df.to_csv(os.path.join(OUTPUT_ROOT, "calculated_metrics.csv"), index=False)
    print(f"\nSaved summary of {len(summary_data)} experiments to {OUTPUT_ROOT}")