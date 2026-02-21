import maya.cmds as cmds
from maya.api import OpenMaya as om
import os
import sys
import tempfile
import subprocess
import struct
import numpy as np

# ======================================================================================
# CONFIGURATION
# ======================================================================================
EXTERNAL_PYTHON = r"C:\Users\janr\Documents\MagCode\.venv_py310\Scripts\python.exe"
RESULTS_ROOT = r"D:\ClothSim\Results"
X_SHIFT_STEP = 1.5  # 1.5 meters spacing between methods

# NOTE: "Maya" must remain the first item so it is loaded and used as the reference!
MODELS = {
    "Maya": "Maya",
    "TailorNet": "TailorNet",
    "CCraft": "ccraft",
    "HOOD": "hood"
}

# Example to load
SEQ_NUM = "01"
SEQ_IDX = "01"
GENDER = "male"
GARMENT = "t-shirt"
CLOTH_TYPE = "cotton"

# ======================================================================================
# CUSTOM PLY PARSER
# ======================================================================================
_SCALAR_FMT = {
    "char":   ("b", 1), "int8": ("b", 1),
    "uchar":  ("B", 1), "uint8":("B", 1),
    "short":  ("h", 2), "int16":("h", 2),
    "ushort": ("H", 2), "uint16":("H", 2),
    "int":    ("i", 4), "int32":("i", 4),
    "uint":   ("I", 4), "uint32":("I", 4),
    "float":  ("f", 4), "float32":("f", 4),
    "double": ("d", 8), "float64":("d", 8),
}

_FACE_LIST_NAMES = ("vertex_indices", "vertex_index", "indices")

def _read_line(f):
    b = f.readline()
    if not b: return None
    return b.decode("ascii", errors="replace").strip()

def _parse_header(f):
    if _read_line(f) != "ply": raise RuntimeError("Not a PLY file.")
    fmt = None; elements = []; current = None
    while True:
        line = _read_line(f)
        if line is None: raise RuntimeError("Unexpected EOF.")
        if line == "end_header": break
        if not line or line.startswith("comment"): continue
        parts = line.split()
        head = parts[0].lower()
        if head == "format": fmt = parts[1].lower()
        elif head == "element":
            if current: elements.append(current)
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}
        elif head == "property":
            if parts[1].lower() == "list":
                current["properties"].append({"kind": "list", "count_type": parts[2].lower(), "item_type": parts[3].lower(), "name": parts[4]})
            else:
                current["properties"].append({"kind": "scalar", "type": parts[1].lower(), "name": parts[2]})
    if current: elements.append(current)
    return elements

def _unpack(f, fmt):
    n = struct.calcsize(fmt); b = f.read(n)
    return struct.unpack(fmt, b)

def _read_scalar(f, t):
    return _unpack(f, "<" + _SCALAR_FMT[t][0])[0]

def _skip_scalar(f, t):
    f.read(_SCALAR_FMT[t][1])

def _triangulate_faces(faces):
    counts, connects = [], []
    for face in faces:
        if len(face) < 3: continue
        if len(face) == 3: counts.append(3); connects.extend(face)
        else:
            v0 = face[0]
            for k in range(1, len(face) - 1):
                counts.append(3); connects.extend([v0, face[k], face[k + 1]])
    return counts, connects

def import_ply_binary_le_as_mesh(path):
    verts, faces = [], []
    with open(path, "rb") as f:
        elements = _parse_header(f)
        vertex_elem = next((e for e in elements if e["name"] == "vertex"), None)
        vprops = vertex_elem["properties"]
        ix, iy, iz = (next(i for i, p in enumerate(vprops) if p["name"] == axis) for axis in ["x", "y", "z"])
        
        for elem in elements:
            ename, ecount, props = elem["name"], elem["count"], elem["properties"]
            if ename == "vertex":
                for _ in range(ecount):
                    row = []
                    for p in props:
                        if p["kind"] == "scalar": row.append(_read_scalar(f, p["type"]))
                        else:
                            ct = int(_read_scalar(f, p["count_type"]))
                            for __ in range(ct): _skip_scalar(f, p["item_type"])
                            row.append(None)
                    verts.append((float(row[ix]), float(row[iy]), float(row[iz])))
            elif ename == "face":
                list_idx = next((i for i, p in enumerate(props) if p["kind"] == "list" and p["name"] in _FACE_LIST_NAMES), None)
                for _ in range(ecount):
                    face_inds = None
                    for i, p in enumerate(props):
                        if p["kind"] == "scalar": _skip_scalar(f, p["type"])
                        else:
                            ct = int(_read_scalar(f, p["count_type"]))
                            if i == list_idx: face_inds = [int(_read_scalar(f, p["item_type"])) for __ in range(ct)]
                            else:
                                for __ in range(ct): _skip_scalar(f, p["item_type"])
                    if face_inds: faces.append(face_inds)
            else:
                for _ in range(ecount):
                    for p in props:
                        if p["kind"] == "scalar": _skip_scalar(f, p["type"])
                        else:
                            ct = int(_read_scalar(f, p["count_type"]))
                            for __ in range(ct): _skip_scalar(f, p["item_type"])

    points = [om.MPoint(x, y, z) for (x, y, z) in verts]
    counts, connects = _triangulate_faces(faces)
    
    mesh_fn = om.MFnMesh()
    
    arr_counts = om.MIntArray(); [arr_counts.append(int(c)) for c in counts]
    arr_connects = om.MIntArray(); [arr_connects.append(int(c)) for c in connects]
    
    created_obj = mesh_fn.create(points, arr_counts, arr_connects)
    
    xform_obj = created_obj if created_obj.hasFn(om.MFn.kTransform) else om.MFnDagNode(created_obj).parent(0)
    xform_path = om.MFnDagNode(xform_obj).fullPathName()
    
    shapes = cmds.listRelatives(xform_path, shapes=True, fullPath=True)
    if shapes: cmds.setAttr(f"{shapes[0]}.displayColors", 0)
    
    cmds.makeIdentity(xform_path, apply=True, t=1, r=1, s=1, n=0)
    return xform_path

# ======================================================================================
# EXTERNAL PKL TO OBJ EXTRACTOR
# ======================================================================================
_PKL_EXTRACTOR_CODE = r"""
import sys
import pickle
import numpy as np

pkl_path = sys.argv[1]
out_obj = sys.argv[2]
target_type = sys.argv[3] # 'cloth' or 'body'
frame_idx = int(sys.argv[4]) # Index of frame to extract

try:
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
        
    verts, faces = None, None
    
    if isinstance(data, dict):
        if target_type == 'cloth':
            verts = data.get('pred', data.get('pred_pos', data.get('vertices')))
            faces = data.get('cloth_faces', data.get('faces'))
        else: # body
            verts = data.get('obstacle', data.get('body', data.get('smpl_verts')))
            faces = data.get('obstacle_faces', data.get('body_faces', data.get('faces')))

    if verts is None:
        print(f"Could not find valid '{target_type}' vertex array in PKL.")
        sys.exit(1)
        
    if len(verts.shape) == 3:
        max_idx = verts.shape[0] - 1
        safe_idx = min(frame_idx, max_idx)
        verts = verts[safe_idx]
        
    if hasattr(verts, 'detach'): verts = verts.detach().cpu().numpy()
    if faces is not None and hasattr(faces, 'detach'): faces = faces.detach().cpu().numpy()
        
    with open(out_obj, 'w') as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        if faces is not None:
            for face in faces:
                f.write(f"f {int(face[0])+1} {int(face[1])+1} {int(face[2])+1}\n")
                
    sys.exit(0)
except Exception as e:
    print(f"Error extracting PKL: {e}")
    sys.exit(1)
"""

# ======================================================================================
# HELPER: MESH CREATION & MATH ALIGNMENT
# ======================================================================================
def _clear_scene_keep_cameras():
    default_cams = {"persp", "top", "front", "side"}
    transforms = cmds.ls(type="transform") or []
    to_delete = [t for t in transforms if t not in default_cams]
    if to_delete: cmds.delete(to_delete)

def _import_obj_or_ply(filepath):
    if not os.path.isfile(filepath): return None
        
    if filepath.endswith('.obj'):
        before = set(cmds.ls(type="transform", long=True) or [])
        cmds.file(filepath, i=True, type="OBJ", ignoreVersion=True, options="mo=1")
        after = set(cmds.ls(type="transform", long=True) or [])
        new = list(after - before)
        if not new: return None
        
        x = new[0]
        for t in new:
            if cmds.listRelatives(t, shapes=True, type="mesh"):
                x = t; break
                
        cmds.makeIdentity(x, apply=True, t=1, r=1, s=1, n=0)
        return x
        
    elif filepath.endswith('.ply'):
        return import_ply_binary_le_as_mesh(filepath)

def _load_mesh_from_pkl_via_bridge(filepath, target_type, frame_idx):
    if not os.path.isfile(filepath): return None
        
    tmp_dir = tempfile.gettempdir()
    extractor_py = os.path.join(tmp_dir, "pkl_extractor.py")
    temp_obj = os.path.join(tmp_dir, f"temp_{target_type}_ext.obj")
    
    with open(extractor_py, "w") as f:
        f.write(_PKL_EXTRACTOR_CODE)
        
    cmd = [EXTERNAL_PYTHON, extractor_py, filepath, temp_obj, target_type, str(frame_idx)]
    p = subprocess.run(cmd, capture_output=True, text=True)
    
    if p.returncode != 0:
        print(f"[ERROR] Failed to extract {target_type} from {filepath}:\n{p.stderr}\n{p.stdout}")
        return None
        
    if os.path.exists(temp_obj):
        imported_node = _import_obj_or_ply(temp_obj)
        os.remove(temp_obj)
        return imported_node
    return None

def _assign_lambert(xform, color, name_suffix):
    if not xform: return
    shader_name = f"mat_{name_suffix}"
    if not cmds.objExists(shader_name):
        shader = cmds.shadingNode("lambert", asShader=True, name=shader_name)
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=f"{shader_name}SG")
        cmds.connectAttr(f"{shader}.outColor", f"{sg}.surfaceShader", force=True)
        cmds.setAttr(f"{shader}.color", *color, type="double3")
    else:
        sg = f"{shader_name}SG"
    cmds.sets(xform, e=True, forceElement=sg)

def _apply_rotation_fix(xforms, rx=0.0, ry=0.0, rz=0.0):
    for x in xforms:
        if x and cmds.objExists(x):
            cmds.rotate(rx, ry, rz, x, r=True, os=True, fo=True)

# --- RIGID ALIGNMENT FUNCTIONS (KABSCH ALGORITHM) ---
def _get_mesh_vertices_numpy(transform_node):
    """Extracts world-space vertices of a mesh to a numpy array."""
    sel = om.MSelectionList()
    sel.add(transform_node)
    dag_path = sel.getDagPath(0)
    if dag_path.apiType() != om.MFn.kMesh:
        dag_path.extendToShape()
        
    mesh_fn = om.MFnMesh(dag_path)
    pts = mesh_fn.getPoints(om.MSpace.kWorld)
    
    out = np.zeros((len(pts), 3), dtype=np.float32)
    for i in range(len(pts)):
        out[i] = [pts[i].x, pts[i].y, pts[i].z]
    return out

def calculate_alignment_matrix(A, B):
    """
    Finds the optimal Rotation & Translation to align point cloud A to B.
    Accepts two numpy arrays.
    Returns a 16-element flat list ready for Maya's cmds.xform(matrix=...)
    """
    if A.shape[0] != B.shape[0]:
        print(f"[WARNING] Vertex count mismatch ({A.shape[0]} vs {B.shape[0]}). Skipping math alignment.")
        return [1,0,0,0, 0,1,0,0, 0,0,1,0, 0,0,0,1] # Identity matrix

    # 1. Find Centroids
    centroid_A = np.mean(A, axis=0)
    centroid_B = np.mean(B, axis=0)

    # 2. Center the point clouds
    AA = A - centroid_A
    BB = B - centroid_B

    # 3. Calculate Covariance Matrix & SVD
    H = np.dot(AA.T, BB)
    U, S, Vt = np.linalg.svd(H)
    
    # 4. Calculate Rotation
    R = np.dot(Vt.T, U.T)

    # Special reflection case handling
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = np.dot(Vt.T, U.T)

    # 5. Calculate Translation
    t = centroid_B - np.dot(R, centroid_A)

    # 6. Format into 4x4 Maya Matrix (Row-Major)
    M = np.eye(4)
    M[:3, :3] = R.T  # Transpose needed for Maya's row-major orientation
    M[3, :3] = t
    
    return M.flatten().tolist()


# ======================================================================================
# MAIN VISUALIZATION LOGIC
# ======================================================================================
def visualize_comparison(frame_indices):
    _clear_scene_keep_cameras()
    
    print(f"\n--- Loading Comparison for {SEQ_NUM}_{SEQ_IDX} {GARMENT} ---")
    
    # We will store the Maya body vertices to use as our alignment reference
    reference_verts = None
    
    for i, (model_display_name, folder_name) in enumerate(MODELS.items()):
        curr_frame_idx = frame_indices.get(model_display_name, 0)
        
        base_dir = os.path.join(RESULTS_ROOT, folder_name, SEQ_NUM, SEQ_IDX, GENDER, GARMENT, CLOTH_TYPE)
        x_offset = i * X_SHIFT_STEP
        print(f"[{model_display_name}] Loading Frame {curr_frame_idx} from: {base_dir}")
        
        body_node, cloth_node = None, None
        
        # --- HARDCODED FILE LOADING ---
        if model_display_name in ["CCraft", "HOOD"]:
            pkl_path = os.path.join(base_dir, "output.pkl")
            if os.path.exists(pkl_path):
                cloth_node = _load_mesh_from_pkl_via_bridge(pkl_path, target_type="cloth", frame_idx=curr_frame_idx)
                body_node = _load_mesh_from_pkl_via_bridge(pkl_path, target_type="body", frame_idx=curr_frame_idx)
            else:
                print(f" -> [WARNING] Missing PKL file: {pkl_path}")
                
        else:
            gar_path = os.path.join(base_dir, "result_ply_files", "pred_gar_{:04d}.ply".format(curr_frame_idx))
            body_path = os.path.join(base_dir, "result_ply_files", "body_{:04d}.ply".format(curr_frame_idx))
            
            if os.path.exists(gar_path): cloth_node = _import_obj_or_ply(gar_path)
            else: print(f" -> [WARNING] Missing cloth file: {gar_path}")
                
            if os.path.exists(body_path): body_node = _import_obj_or_ply(body_path)
            else: print(f" -> [WARNING] Missing body file: {body_path}")
        
        # --- MATERIAL ASSIGNMENT ---
        if cloth_node: _assign_lambert(cloth_node, (0.2, 0.6, 1.0), "ClothBlue")
        if body_node: _assign_lambert(body_node, (0.4, 0.4, 0.4), "BodyGray")
            
        # --- PROCESSING, ALIGNING & GROUPING ---
        nodes_to_group = [n for n in [body_node, cloth_node] if n]
        if nodes_to_group:
            
            # --- APPLY PRELIMINARY TAILORNET FIX BEFORE ALIGNMENT ---
            if model_display_name == "TailorNet":
                # Rotate first so the Procrustes algorithm starts close to the target
                _apply_rotation_fix(nodes_to_group, rz=-90)
                _apply_rotation_fix(nodes_to_group, rx=-90)
                # Bake the transformation so the vertex positions update in world space
                for node in nodes_to_group:
                    cmds.makeIdentity(node, apply=True, t=1, r=1, s=1, n=0)

            align_matrix = None
            
            # If this is Maya, extract and cache its vertices BEFORE grouping
            if model_display_name == "Maya" and body_node:
                reference_verts = _get_mesh_vertices_numpy(body_node)
                
            # If it's NOT Maya, extract its vertices and calculate the transform
            elif model_display_name != "Maya" and reference_verts is not None and body_node:
                print(f" -> Automatically aligning {model_display_name} to Maya reference...")
                src_verts = _get_mesh_vertices_numpy(body_node)
                align_matrix = calculate_alignment_matrix(src_verts, reference_verts)

            # 1. Group the nodes safely
            group_name = cmds.group(nodes_to_group, name=f"GRP_{model_display_name}")

            # 2. Apply the calculated Procrustes transformation to the Group
            if align_matrix:
                cmds.xform(group_name, matrix=align_matrix, worldSpace=True)

            # 3. Apply the X-axis shift to place it in its lineup slot
            cmds.move(x_offset, 0, 0, group_name, relative=True, worldSpace=True)
            
            # 4. Text Label
            text_curves = cmds.textCurves(text=model_display_name, font="Arial|h-13|w400|c0", name=f"TXT_{model_display_name}")[0]
            cmds.move(x_offset, 2.2, 0, text_curves, worldSpace=True)
            cmds.scale(0.1, 0.1, 0.1, text_curves)
        else:
            print(f" -> [MISSING DATA] No meshes loaded.")
            
    cmds.viewFit("persp", all=True)
    print("--- Done ---")

if __name__ == "__main__":
    FRAME_INDICES = {
        "Maya": 10,
        "TailorNet": 10,
        "CCraft": 10,
        "HOOD": 8
    }
    visualize_comparison(FRAME_INDICES)