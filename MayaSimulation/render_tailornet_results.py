from maya.api import OpenMaya as om
import maya.cmds as cmds
import os
import re
import struct
import subprocess
import sys

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

def _clear_scene_keep_cameras():
    """Delete all transforms except default cameras."""
    default_cams = {"persp", "top", "front", "side"}
    transforms = cmds.ls(type="transform") or []
    to_delete = [t for t in transforms if t not in default_cams]
    if to_delete:
        cmds.delete(to_delete)

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
    
def _mint_array(seq):
    # Most compatible across Maya builds: append()
    arr = om.MIntArray()
    for x in seq:
        arr.append(int(x))
    return arr

def _ensure_dir(p):
    if not os.path.isdir(p):
        os.makedirs(p, exist_ok=True)

def _find_frame_indices(results_path, body_prefix="body_", gar_prefix="pred_gar_"):
    """
    Finds frame indices available in results_path by matching body_####.ply and pred_gar_####.ply.
    Returns sorted list of ints.
    """
    body_re = re.compile(rf"^{re.escape(body_prefix)}(\d+)\.ply$")
    gar_re  = re.compile(rf"^{re.escape(gar_prefix)}(\d+)\.ply$")

    body_frames = set()
    gar_frames = set()

    for fn in os.listdir(results_path):
        m = body_re.match(fn)
        if m:
            body_frames.add(int(m.group(1)))
            continue
        m = gar_re.match(fn)
        if m:
            gar_frames.add(int(m.group(1)))

    frames = sorted(body_frames.intersection(gar_frames))
    if not frames:
        raise RuntimeError(
            f"No matching frames found in {results_path}. "
            f"Expected {body_prefix}####.ply and {gar_prefix}####.ply"
        )
    return frames

def _setup_camera(cam="persp", fit=True):
    # Make sure persp exists and is renderable
    if not cmds.objExists(cam):
        raise RuntimeError(f"Camera '{cam}' not found.")
    # Optionally frame all after import per frame; we do that in loop if requested
    cmds.lookThru(cam)

def _set_viewport2_defaults():
    """
    Reasonable defaults for viewport playblast.
    (Avoids surprises across machines.)
    """
    try:
        cmds.setAttr("hardwareRenderingGlobals.multiSampleEnable", 1)
    except Exception:
        pass

def _import_frame_meshes(results_path, frame_idx, pad=4,
                         body_color=(0.2, 0.2, 0.2),
                         gar_color=(0.2, 0.6, 1.0),
                         color_mode="material"):
    idx_str = str(frame_idx).zfill(pad)
    body_path = os.path.join(results_path, f"body_{idx_str}.ply")
    gar_path  = os.path.join(results_path, f"pred_gar_{idx_str}.ply")

    # Name per frame so we can delete easily
    gar_xform, _ = import_ply_binary_le_as_mesh(
        gar_path,
        name=f"gar_{idx_str}",
        color=gar_color,
        color_mode=color_mode,
    )
    body_xform, _ = import_ply_binary_le_as_mesh(
        body_path,
        name=f"body_{idx_str}",
        color=body_color,
        color_mode=color_mode,
    )
    return body_xform, gar_xform

def _delete_nodes(nodes):
    nodes = [n for n in nodes if n and cmds.objExists(n)]
    if nodes:
        cmds.delete(nodes)

def render_sequence_to_images_vp2(
    results_path,
    out_dir,
    cam="persp",
    start=None,
    end=None,
    pad=4,
    width=1024,
    height=1024,
    fit_camera_each_frame=False,
    image_ext="png",
):
    """
    Renders using Viewport 2.0 via playblast to an image sequence.
    Produces: out_dir/frame_0000.png, frame_0001.png, ...
    """
    results_path = os.path.abspath(results_path)
    out_dir = os.path.abspath(out_dir)
    _ensure_dir(out_dir)

    frames = _find_frame_indices(results_path)
    if start is None: start = frames[0]
    if end is None: end = frames[-1]
    frames = [f for f in frames if start <= f <= end]
    if not frames:
        raise RuntimeError("No frames in selected range.")

    # Scene prep
    _clear_scene_keep_cameras()
    _setup_camera(cam=cam)
    _set_viewport2_defaults()

    # Render each frame to a still (playblast frame-by-frame)
    # We create a temporary timeline so playblast writes correctly.
    cmds.playbackOptions(min=0, max=len(frames)-1)
    cmds.currentTime(0)

    # We'll import meshes per frame, frame them optionally, then playblast single frame.
    for local_i, frame_idx in enumerate(frames):
        cmds.currentTime(local_i, edit=True)

        # delete previous frame meshes
        _clear_scene_keep_cameras()

        body_node, gar_node = _import_frame_meshes(results_path, frame_idx, pad=pad)

        if fit_camera_each_frame:
            # frame all visible objects in the active view
            try:
                cmds.viewFit(cam, all=True)
            except Exception:
                pass

        print(f"frame_{str(local_i).zfill(pad)}")
        out_path_noext = os.path.join(out_dir, f"frame_{str(local_i).zfill(pad)}")
        print(out_path_noext)
        cmds.playblast(
            format="image",
            # filename=os.path.join(out_dir, "frame_"),
            filename=out_path_noext,  # Maya appends .png/.jpg
            framePadding=4,
            compression=image_ext,
            frame=local_i,
            sequenceTime=False,
            clearCache=True,
            viewer=False,
            showOrnaments=False,
            offScreen=True,
            percent=100,
            widthHeight=(width, height),
            forceOverwrite=True,
        )

    return frames

def _find_ffmpeg():
    # Try PATH
    return "ffmpeg"

def images_to_mp4_ffmpeg(img_dir, out_mp4, fps=30, pad=4, image_ext="png"):
    """
    Assembles img_dir/frame_0000.png ... into out_mp4 using ffmpeg.
    Requires ffmpeg installed and accessible from Maya's environment.
    """
    img_dir = os.path.abspath(img_dir)
    out_mp4 = os.path.abspath(out_mp4)
    _ensure_dir(os.path.dirname(out_mp4))

    pattern = os.path.join(img_dir, f"frame_%0{pad}d.0000.{image_ext}")
    ffmpeg = _find_ffmpeg()

    cmd = [
        ffmpeg,
        "-y",
        "-framerate", str(int(fps)),
        "-i", pattern,
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        "-crf", "18",
        out_mp4,
    ]

    # Run ffmpeg
    try:
        p = subprocess.run(cmd, capture_output=True, text=True)
    except FileNotFoundError:
        raise RuntimeError(
            "ffmpeg not found. Install ffmpeg and ensure Maya can access it via PATH.\n"
            "On Windows: add ffmpeg/bin to PATH, then restart Maya.\n"
            "On macOS: brew install ffmpeg\n"
            "On Linux: apt/yum/pacman install ffmpeg"
        )

    if p.returncode != 0:
        raise RuntimeError(f"ffmpeg failed:\nSTDOUT:\n{p.stdout}\nSTDERR:\n{p.stderr}")

    return out_mp4

# ---------------------------------------------------------------------
# Example usage
# ---------------------------------------------------------------------
results_path = "/Users/jan.rojc/Documents/MagCode/Data/Results/TailorNet/"
if not os.path.exists(results_path):
    results_path = "D:/ClothSim/Results/TailorNet/"

img_out = os.path.join(results_path, "_image_renders")
mp4_out = os.path.join(results_path, "_tailornet_preview.mp4")
if os.path.exists(mp4_out):
    print("Removing existing mp4:", mp4_out)
    os.remove(mp4_out)

# Render to images
render_sequence_to_images_vp2(
    results_path=results_path,
    out_dir=img_out,
    cam="persp",
    start=None,
    end=None,
    pad=4,
    width=1024,
    height=1024,
    fit_camera_each_frame=False,  # set True if framing is off
    image_ext="png",
)

# Assemble to video
images_to_mp4_ffmpeg(
    img_dir=img_out,
    out_mp4=mp4_out,
    fps=10,       # match your intended playback
    pad=4,
    image_ext="png",
)

print("Done:", mp4_out)