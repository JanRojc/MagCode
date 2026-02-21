import maya.cmds as cmds
import os
import sys
import tempfile
import subprocess
import struct
import numpy as np
import math

# ======================================================================================
# CONFIGURATION
# ======================================================================================
EXTERNAL_PYTHON = r"C:\Users\janr\Documents\MagCode\.venv_py310\Scripts\python.exe"
RESULTS_ROOT = r"D:\ClothSim\Results"
X_SHIFT_STEP = 1.5  # 1.5 meters spacing between methods

MODELS = {
    "Maya": "Maya",
    "TailorNet": "TailorNet",
    "CCraft": "ccraft",
    "HOOD": "hood"
}

SEQ_NUM = "01"
SEQ_IDX = "01"
GENDER = "male"
GARMENT = "t-shirt"
CLOTH_TYPE = "cotton"

# ======================================================================================
# CUSTOM PLY PARSER (Minified)
# ======================================================================================
_SCALAR_FMT = {
    "char": ("b", 1), "int8": ("b", 1), "uchar": ("B", 1), "uint8": ("B", 1),
    "short": ("h", 2), "int16": ("h", 2), "ushort": ("H", 2), "uint16": ("H", 2),
    "int": ("i", 4), "int32": ("i", 4), "uint": ("I", 4), "uint32": ("I", 4),
    "float": ("f", 4), "float32": ("f", 4), "double": ("d", 8), "float64": ("d", 8),
}

def import_ply_binary_le_as_mesh(path):
    verts, faces = [], []
    with open(path, "rb") as f:
        elements = []
        fmt = None; current = None
        while True:
            line = f.readline()
            if not line: break
            line = line.decode("ascii", errors="replace").strip()
            if line == "end_header": break
            if not line or line.startswith("comment"): continue
            parts = line.split()
            if parts[0].lower() == "format": fmt = parts[1].lower()
            elif parts[0].lower() == "element":
                if current: elements.append(current)
                current = {"name": parts[1], "count": int(parts[2]), "properties": []}
            elif parts[0].lower() == "property":
                if parts[1].lower() == "list":
                    current["properties"].append({"kind": "list", "count_type": parts[2].lower(), "item_type": parts[3].lower(), "name": parts[4]})
                else:
                    current["properties"].append({"kind": "scalar", "type": parts[1].lower(), "name": parts[2]})
        if current: elements.append(current)

        vprops = next((e["properties"] for e in elements if e["name"] == "vertex"), None)
        ix, iy, iz = (next(i for i, p in enumerate(vprops) if p["name"] == axis) for axis in ["x", "y", "z"])
        
        for elem in elements:
            for _ in range(elem["count"]):
                if elem["name"] == "vertex":
                    row = []
                    for p in elem["properties"]:
                        if p["kind"] == "scalar":
                            row.append(struct.unpack("<" + _SCALAR_FMT[p["type"]][0], f.read(_SCALAR_FMT[p["type"]][1]))[0])
                        else:
                            ct = struct.unpack("<" + _SCALAR_FMT[p["count_type"]][0], f.read(_SCALAR_FMT[p["count_type"]][1]))[0]
                            f.read(_SCALAR_FMT[p["item_type"]][1] * ct)
                            row.append(None)
                    verts.append((float(row[ix]), float(row[iy]), float(row[iz])))
                elif elem["name"] == "face":
                    face_inds = None
                    list_idx = next((i for i, p in enumerate(elem["properties"]) if p["kind"] == "list"), None)
                    for i, p in enumerate(elem["properties"]):
                        if p["kind"] == "scalar": f.read(_SCALAR_FMT[p["type"]][1])
                        else:
                            ct = struct.unpack("<" + _SCALAR_FMT[p["count_type"]][0], f.read(_SCALAR_FMT[p["count_type"]][1]))[0]
                            if i == list_idx:
                                face_inds = [struct.unpack("<" + _SCALAR_FMT[p["item_type"]][0], f.read(_SCALAR_FMT[p["item_type"]][1]))[0] for __ in range(ct)]
                            else: f.read(_SCALAR_FMT[p["item_type"]][1] * ct)
                    if face_inds: faces.append(face_inds)

    flat_pts, flat_counts, flat_connects = [], [], []
    for x, y, z in verts: flat_pts.extend([x, y, z])
    for face in faces:
        if len(face) == 3:
            flat_counts.append(3); flat_connects.extend(face)
        elif len(face) > 3:
            v0 = face[0]
            for k in range(1, len(face) - 1):
                flat_counts.append(3); flat_connects.extend([v0, face[k], face[k + 1]])

    # Build via cmds to avoid OpenMaya naming issues
    mesh_node = cmds.polyCreateFacet(p=verts)[0]
    cmds.delete(mesh_node) # Delete empty placeholder, use robust creation
    
    # Actually, polyCreateFacet is slow for many polys. Let's write it to a temp OBJ and import via cmds.
    # This completely bypasses OpenMaya bugs in your scene.
    tmp_obj = os.path.join(tempfile.gettempdir(), "temp_ply_converted.obj")
    with open(tmp_obj, 'w') as f_obj:
        for x,y,z in verts: f_obj.write(f"v {x} {y} {z}\n")
        for face in faces: 
            f_obj.write("f " + " ".join([str(idx+1) for idx in face]) + "\n")
            
    before = set(cmds.ls(type="transform", long=True) or [])
    cmds.file(tmp_obj, i=True, type="OBJ", ignoreVersion=True, options="mo=1")
    after = set(cmds.ls(type="transform", long=True) or [])
    new = list(after - before)
    if os.path.exists(tmp_obj): os.remove(tmp_obj)
    
    return new[0] if new else None


# ======================================================================================
# EXTERNAL PKL TO OBJ EXTRACTOR
# ======================================================================================
_PKL_EXTRACTOR_CODE = r"""
import sys
import pickle
import numpy as np

pkl_path = sys.argv[1]
out_obj = sys.argv[2]
target_type = sys.argv[3]
frame_idx = int(sys.argv[4])

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    verts, faces = None, None
    if isinstance(data, dict):
        if target_type == 'cloth':
            verts = data.get('pred', data.get('pred_pos', data.get('vertices')))
            faces = data.get('cloth_faces', data.get('faces'))
        else:
            verts = data.get('obstacle', data.get('body', data.get('smpl_verts')))
            faces = data.get('obstacle_faces', data.get('body_faces', data.get('faces')))

    if verts is None: sys.exit(1)
        
    if len(verts.shape) == 3:
        max_idx = verts.shape[0] - 1
        verts = verts[min(frame_idx, max_idx)]
        
    if hasattr(verts, 'detach'): verts = verts.detach().cpu().numpy()
    if faces is not None and hasattr(faces, 'detach'): faces = faces.detach().cpu().numpy()
        
    with open(out_obj, 'w') as f:
        for v in verts: f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if faces is not None:
            for face in faces: f.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")
    sys.exit(0)
except Exception:
    sys.exit(1)
"""

# ======================================================================================
# HELPER FUNCTIONS
# ======================================================================================
def _clear_scene_keep_cameras():
    defaults = {"persp", "top", "front", "side"}
    to_delete = [t for t in (cmds.ls(type="transform") or []) if t not in defaults]
    if to_delete: cmds.delete(to_delete)

def _import_obj_or_ply(filepath):
    if not os.path.isfile(filepath): return None
    if filepath.endswith('.obj'):
        before = set(cmds.ls(type="transform", long=True) or [])
        cmds.file(filepath, i=True, type="OBJ", ignoreVersion=True, options="mo=1")
        new = list(set(cmds.ls(type="transform", long=True) or []) - before)
        return new[0] if new else None
    elif filepath.endswith('.ply'):
        return import_ply_binary_le_as_mesh(filepath)

def _load_mesh_from_pkl_via_bridge(filepath, target_type, frame_idx):
    if not os.path.isfile(filepath): return None
    tmp_dir = tempfile.gettempdir()
    extractor_py = os.path.join(tmp_dir, "pkl_extractor.py")
    temp_obj = os.path.join(tmp_dir, f"temp_{target_type}_ext.obj")
    with open(extractor_py, "w") as f: f.write(_PKL_EXTRACTOR_CODE)
    subprocess.run([EXTERNAL_PYTHON, extractor_py, filepath, temp_obj, target_type, str(frame_idx)], capture_output=True)
    if os.path.exists(temp_obj):
        node = _import_obj_or_ply(temp_obj)
        os.remove(temp_obj)
        return node
    return None

def _assign_lambert(xform, color, name_suffix):
    if not xform: return
    shader_name = f"mat_{name_suffix}"
    if not cmds.objExists(shader_name):
        shader = cmds.shadingNode("lambert", asShader=True, name=shader_name)
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=f"{shader_name}SG")
        cmds.connectAttr(f"{shader}.outColor", f"{sg}.surfaceShader", force=True)
        cmds.setAttr(f"{shader}.color", *color, type="double3")
    else: sg = f"{shader_name}SG"
    cmds.sets(xform, e=True, forceElement=sg)

# --- BULLETPROOF VERTEX EXTRACTOR ---
def _get_mesh_vertices_numpy(transform_node):
    """Safely extracts vertices using cmds, avoiding OpenMaya API crashes."""
    if not transform_node or not cmds.objExists(transform_node): return None
    
    # Flat list: [x1, y1, z1, x2, y2, z2, ...]
    vtx_flat = cmds.xform(f"{transform_node}.vtx[*]", query=True, translation=True, worldSpace=True)
    
    # Convert to Nx3 Numpy array
    out = np.array(vtx_flat, dtype=np.float32).reshape(-1, 3)
    return out

# --- FAST ICP ALGORITHM (SHAPE ALIGNMENT) ---
def calculate_fast_icp_alignment(source_pts, target_pts, max_iters=15, sample_size=500):
    """
    Lightning-fast Iterative Closest Point using random downsampling.
    Solves the 30-second delay and ignores topology/resolution differences.
    """
    if source_pts is None or target_pts is None: return None
    
    # 1. Initial Centroid Translation (Calculated on FULL meshes for high accuracy)
    mean_src_full = np.mean(source_pts, axis=0)
    mean_tgt_full = np.mean(target_pts, axis=0)
    init_t = mean_tgt_full - mean_src_full
    
    # Apply initial translation to source points
    src_full_shifted = source_pts + init_t
    
    # 2. Downsample for ultra-fast rotation matching
    np.random.seed(42) # Keep results consistent
    
    # Pick 500 random points (or less if the mesh is somehow tiny)
    n_src = min(sample_size, src_full_shifted.shape[0])
    n_tgt = min(sample_size, target_pts.shape[0])
    
    src = src_full_shifted[np.random.choice(src_full_shifted.shape[0], n_src, replace=False)]
    tgt = target_pts[np.random.choice(target_pts.shape[0], n_tgt, replace=False)]
    
    total_R = np.eye(3)
    total_t = np.copy(init_t)
    
    for _ in range(max_iters):
        # Broadcasting difference (Fast nearest neighbor)
        diff = src[:, None, :] - tgt[None, :, :]
        dists = np.sum(diff**2, axis=-1)
        indices = np.argmin(dists, axis=1)
        
        matched_tgt = tgt[indices]
        
        # Kabsch Algorithm on the 500 matched pairs
        mean_src = np.mean(src, axis=0)
        mean_matched = np.mean(matched_tgt, axis=0)
        
        src_centered = src - mean_src
        matched_centered = matched_tgt - mean_matched
        
        H = np.dot(src_centered.T, matched_centered)
        U, S, Vt = np.linalg.svd(H)
        R = np.dot(Vt.T, U.T)
        
        # Prevent mirroring
        if np.linalg.det(R) < 0:
            Vt[2, :] *= -1
            R = np.dot(Vt.T, U.T)
            
        t = mean_matched - np.dot(R, mean_src)
        
        # Accumulate transform
        src = np.dot(src, R.T) + t
        total_R = np.dot(R, total_R)
        total_t = np.dot(R, total_t) + t

    return [
        total_R[0,0], total_R[1,0], total_R[2,0], 0.0,
        total_R[0,1], total_R[1,1], total_R[2,1], 0.0,
        total_R[0,2], total_R[1,2], total_R[2,2], 0.0,
        total_t[0],   total_t[1],   total_t[2],   1.0
    ]

# ======================================================================================
# MAIN VISUALIZATION
# ======================================================================================
def visualize_comparison(frame_indices):
    _clear_scene_keep_cameras()
    print(f"\n--- Loading Comparison for {SEQ_NUM}_{SEQ_IDX} {GARMENT} ---")
    
    reference_verts = None
    
    for i, (model_display_name, folder_name) in enumerate(MODELS.items()):
        curr_frame_idx = frame_indices.get(model_display_name, 0)
        base_dir = os.path.join(RESULTS_ROOT, folder_name, SEQ_NUM, SEQ_IDX, GENDER, GARMENT, CLOTH_TYPE)
        x_offset = i * X_SHIFT_STEP
        print(f"[{model_display_name}] Loading...")
        
        body_node, cloth_node = None, None
        
        # LOAD
        if model_display_name in ["CCraft", "HOOD"]:
            pkl_path = os.path.join(base_dir, "output.pkl")
            cloth_node = _load_mesh_from_pkl_via_bridge(pkl_path, "cloth", curr_frame_idx)
            body_node = _load_mesh_from_pkl_via_bridge(pkl_path, "body", curr_frame_idx)
        else:
            gar_path = os.path.join(base_dir, "result_ply_files", f"pred_gar_{curr_frame_idx:04d}.ply")
            body_path = os.path.join(base_dir, "result_ply_files", f"body_{curr_frame_idx:04d}.ply")
            cloth_node = _import_obj_or_ply(gar_path)
            body_node = _import_obj_or_ply(body_path)

        # MATERIALS
        _assign_lambert(cloth_node, (0.2, 0.6, 1.0), "ClothBlue")
        _assign_lambert(body_node, (0.4, 0.4, 0.4), "BodyGray")
            
        nodes_to_group = [n for n in [body_node, cloth_node] if n]
        if nodes_to_group:
            
            # PRE-ROTATION (Gives ICP a head start so it doesn't align sideways)
            if model_display_name == "TailorNet":
                for node in nodes_to_group: cmds.rotate(-90, 0, -90, node, r=True, os=True, fo=True)
                for node in nodes_to_group: cmds.makeIdentity(node, apply=True, t=1, r=1, s=1, n=0)

            # ICP ALIGNMENT
            if model_display_name == "Maya" and body_node:
                reference_verts = _get_mesh_vertices_numpy(body_node)
                
            elif model_display_name != "Maya" and reference_verts is not None and body_node:
                src_verts = _get_mesh_vertices_numpy(body_node)
                align_matrix = calculate_fast_icp_alignment(src_verts, reference_verts)

                if align_matrix:
                    import maya.api.OpenMaya as om
                    tm = om.MTransformationMatrix(om.MMatrix(align_matrix))
                    rot = tm.rotation(asQuaternion=False)
                    rx, ry, rz = math.degrees(rot.x), math.degrees(rot.y), math.degrees(rot.z)
                    tx, ty, tz = tm.translation(om.MSpace.kWorld)
                    
                    print(f"   -> ICP aligned: Move [{tx:.3f}, {ty:.3f}, {tz:.3f}], Rot [{rx:.1f}°, {ry:.1f}°, {rz:.1f}°]")
                    for node in nodes_to_group:
                        cmds.xform(node, translation=[tx, ty, tz], rotation=[rx, ry, rz], worldSpace=True, relative=True)

            # GROUP & SHIFT
            group_name = cmds.group(nodes_to_group, name=f"GRP_{model_display_name}")
            cmds.move(x_offset, 0, 0, group_name, relative=True, worldSpace=True)
            
            # LABEL
            txt = cmds.textCurves(text=model_display_name, font="Arial|h-13|w400|c0")[0]
            cmds.move(x_offset, 2.2, 0, txt, worldSpace=True)
            cmds.scale(0.1, 0.1, 0.1, txt)
            
    cmds.viewFit("persp", all=True)

if __name__ == "__main__":
    FRAME_INDICES = { "Maya": 10, "TailorNet": 10, "CCraft": 10, "HOOD": 8 }
    visualize_comparison(FRAME_INDICES)