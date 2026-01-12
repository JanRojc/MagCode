from maya.api import OpenMaya as om
import maya.cmds as cmds
import os
import struct

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
    if not b:
        return None
    return b.decode("ascii", errors="replace").strip()


def _parse_header(f):
    if _read_line(f) != "ply":
        raise RuntimeError("Not a PLY file (missing 'ply').")

    fmt = None
    elements = []
    current = None

    while True:
        line = _read_line(f)
        if line is None:
            raise RuntimeError("Unexpected EOF while reading PLY header.")
        if line == "end_header":
            break
        if not line or line.startswith("comment"):
            continue

        parts = line.split()
        head = parts[0].lower()

        if head == "format":
            # format binary_little_endian 1.0
            fmt = parts[1].lower()

        elif head == "element":
            # element vertex 123
            if current:
                elements.append(current)
            current = {"name": parts[1], "count": int(parts[2]), "properties": []}

        elif head == "property":
            if current is None:
                raise RuntimeError("PLY header has 'property' before any 'element'.")

            if parts[1].lower() == "list":
                # property list uchar int vertex_indices
                current["properties"].append({
                    "kind": "list",
                    "count_type": parts[2].lower(),
                    "item_type": parts[3].lower(),
                    "name": parts[4],
                })
            else:
                # property float x
                current["properties"].append({
                    "kind": "scalar",
                    "type": parts[1].lower(),
                    "name": parts[2],
                })

    if current:
        elements.append(current)

    if fmt != "binary_little_endian":
        raise RuntimeError(f"Unsupported PLY format '{fmt}'. Need binary_little_endian.")

    return elements


def _unpack(f, fmt):
    n = struct.calcsize(fmt)
    b = f.read(n)
    if len(b) != n:
        raise RuntimeError("Unexpected EOF while reading binary PLY data.")
    return struct.unpack(fmt, b)


def _read_scalar(f, t):
    if t not in _SCALAR_FMT:
        raise RuntimeError(f"Unsupported scalar type '{t}'.")
    code, _ = _SCALAR_FMT[t]
    return _unpack(f, "<" + code)[0]


def _skip_scalar(f, t):
    if t not in _SCALAR_FMT:
        raise RuntimeError(f"Unsupported scalar type '{t}'.")
    _, nbytes = _SCALAR_FMT[t]
    b = f.read(nbytes)
    if len(b) != nbytes:
        raise RuntimeError("Unexpected EOF while skipping binary PLY data.")


def _triangulate_faces(faces):
    counts, connects = [], []
    for face in faces:
        if len(face) < 3:
            continue
        if len(face) == 3:
            counts.append(3)
            connects.extend(face)
        else:
            v0 = face[0]
            for k in range(1, len(face) - 1):
                counts.append(3)
                connects.extend([v0, face[k], face[k + 1]])
    return counts, connects


def _mint_array(seq):
    # Most compatible across Maya builds: append()
    arr = om.MIntArray()
    for x in seq:
        arr.append(int(x))
    return arr


def _sanitize_color(color):
    if color is None:
        return None
    if not (isinstance(color, (tuple, list)) and len(color) == 3):
        raise RuntimeError("color must be a tuple/list of 3 floats in 0..1, e.g. (1.0, 0.2, 0.2)")
    r, g, b = (float(color[0]), float(color[1]), float(color[2]))
    # clamp
    r = max(0.0, min(1.0, r))
    g = max(0.0, min(1.0, g))
    b = max(0.0, min(1.0, b))
    return (r, g, b)


def _apply_color(transform, shape, color, mode):
    """
    Apply either viewport override color or material color.
    transform: transform node name
    shape: shape node name
    color: (r,g,b) 0..1
    mode: "material" or "object"
    """
    if not color:
        return

    r, g, b = color

    if mode == "object":
        # Viewport object color (override)
        cmds.setAttr(f"{shape}.overrideEnabled", 1)
        cmds.setAttr(f"{shape}.overrideRGBColors", 1)
        cmds.setAttr(f"{shape}.overrideColorRGB", r, g, b)

    elif mode == "material":
        # Minimal renderer-agnostic shading network
        shader = cmds.shadingNode("lambert", asShader=True, name=f"{transform}_mat")
        sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True, name=f"{shader}SG")
        cmds.connectAttr(f"{shader}.outColor", f"{sg}.surfaceShader", force=True)

        cmds.setAttr(f"{shader}.color", r, g, b, type="double3")
        cmds.sets(shape, e=True, forceElement=sg)

    else:
        raise RuntimeError(f"Unknown color_mode '{mode}'. Use 'material' or 'object'.")


def import_ply_binary_le_as_mesh(
    path,
    name=None,
    triangulate=True,
    color=None,              # (r,g,b) in 0..1
    color_mode="material",   # "material" | "object"
):
    """
    Import a binary_little_endian PLY into Maya using OpenMaya (no plyTranslator plugin).

    Args:
        path (str): path to .ply
        name (str|None): transform name
        triangulate (bool): fan-triangulate n-gons
        color ((r,g,b)|None): optional color in 0..1
        color_mode (str): "material" or "object"

    Returns:
        (transform_name, shape_name)
    """
    if not os.path.isfile(path):
        raise RuntimeError(f"File not found: {path}")

    color = _sanitize_color(color)

    verts = []
    faces = []

    with open(path, "rb") as f:
        elements = _parse_header(f)

        vertex_elem = next((e for e in elements if e["name"] == "vertex"), None)
        if not vertex_elem:
            raise RuntimeError("PLY has no 'vertex' element.")

        vprops = vertex_elem["properties"]

        # prefer x/y/z by name; fallback to first 3 scalar properties
        def _prop_index_scalar(prop_name):
            for i, p in enumerate(vprops):
                if p["kind"] == "scalar" and p["name"] == prop_name:
                    return i
            return None

        ix, iy, iz = _prop_index_scalar("x"), _prop_index_scalar("y"), _prop_index_scalar("z")
        if ix is None or iy is None or iz is None:
            scalar_idxs = [i for i, p in enumerate(vprops) if p["kind"] == "scalar"]
            if len(scalar_idxs) < 3:
                raise RuntimeError("Vertex element lacks 3 scalar properties for position.")
            ix, iy, iz = scalar_idxs[:3]

        # Read the binary stream in *element order* as declared in the header.
        for elem in elements:
            ename, ecount, props = elem["name"], elem["count"], elem["properties"]

            if ename == "vertex":
                for _ in range(ecount):
                    row = []
                    for p in props:
                        if p["kind"] == "scalar":
                            row.append(_read_scalar(f, p["type"]))
                        else:
                            # Rare on vertex, but skip correctly
                            ct = int(_read_scalar(f, p["count_type"]))
                            for __ in range(ct):
                                _skip_scalar(f, p["item_type"])
                            row.append(None)
                    verts.append((float(row[ix]), float(row[iy]), float(row[iz])))

            elif ename == "face":
                # Find the list property containing vertex indices
                list_idx = None
                for i, p in enumerate(props):
                    if p["kind"] == "list" and p["name"] in _FACE_LIST_NAMES:
                        list_idx = i
                        break
                if list_idx is None:
                    raise RuntimeError("Face element exists, but no supported indices list property found.")

                for _ in range(ecount):
                    face_inds = None
                    for i, p in enumerate(props):
                        if p["kind"] == "scalar":
                            _skip_scalar(f, p["type"])
                        else:
                            ct = int(_read_scalar(f, p["count_type"]))
                            if i == list_idx:
                                inds = [int(_read_scalar(f, p["item_type"])) for __ in range(ct)]
                                face_inds = inds
                            else:
                                for __ in range(ct):
                                    _skip_scalar(f, p["item_type"])
                    if face_inds:
                        faces.append(face_inds)

            else:
                # Unknown element: skip its payload correctly
                for _ in range(ecount):
                    for p in props:
                        if p["kind"] == "scalar":
                            _skip_scalar(f, p["type"])
                        else:
                            ct = int(_read_scalar(f, p["count_type"]))
                            for __ in range(ct):
                                _skip_scalar(f, p["item_type"])

    if not verts:
        raise RuntimeError("PLY contains 0 vertices.")
    if not faces:
        raise RuntimeError("PLY contains 0 faces (or faces couldn't be read).")

    # Sanity-check indices (quick)
    vlen = len(verts)
    for face in faces[:5000]:
        for idx in face:
            if idx < 0 or idx >= vlen:
                raise RuntimeError("Face indices out of range; PLY parse mismatch or corrupt file.")

    # Build mesh
    points = [om.MPoint(x, y, z) for (x, y, z) in verts]

    if triangulate:
        counts, connects = _triangulate_faces(faces)
    else:
        counts = [len(fa) for fa in faces]
        connects = [i for fa in faces for i in fa]

    if not counts:
        raise RuntimeError("No valid polygon data after triangulation.")

    counts_arr = _mint_array(counts)
    connects_arr = _mint_array(connects)

    mesh_fn = om.MFnMesh()
    mesh_obj = mesh_fn.create(points, counts_arr, connects_arr)

    if mesh_obj.isNull():
        raise RuntimeError("MFnMesh.create() failed (null object).")

    # Get transform + shape names
    dag = om.MFnDagNode(mesh_obj)
    shape_name = dag.name()

    xform_obj = dag.parent(0)
    xform_fn = om.MFnDagNode(xform_obj)

    if not name:
        base = os.path.splitext(os.path.basename(path))[0]
        name = base or "plyMesh"

    xform_name = xform_fn.setName(name)

    # Apply optional color
    _apply_color(xform_name, shape_name, color, color_mode)

    # Select it (best effort)
    try:
        sel = om.MSelectionList()
        sel.add(xform_name)
        om.MGlobal.setActiveSelectionList(sel)
    except Exception:
        pass

    return xform_name, shape_name

def _clear_scene_keep_cameras():
    """Delete all transforms except default cameras."""
    default_cams = {"persp", "top", "front", "side"}
    transforms = cmds.ls(type="transform") or []
    to_delete = [t for t in transforms if t not in default_cams]
    if to_delete:
        cmds.delete(to_delete)





# ----------------------------------------------------------------------------------------------------------- #
results_path = "/Users/jan.rojc/Documents/MagCode/Data/Results/TailorNet/"
if not os.path.exists(results_path): results_path = "D:/ClothSim/Results/TailorNet/"
result_index = "0000"

# clear scene
_clear_scene_keep_cameras()

# show garment
import_ply_binary_le_as_mesh(
     results_path+"pred_gar_" + result_index + ".ply",
     name="importedPLY",
     color=(0.2, 0.6, 1.0),
     color_mode="material",
)

# show body
import_ply_binary_le_as_mesh(
     results_path+"body_" + result_index + ".ply",
     name="importedPLY",
     color=(0.2, 0.2, 0.2),
     color_mode="material",
)