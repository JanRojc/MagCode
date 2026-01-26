import maya.cmds as cmds

def spawn_obj(obj_path, clearScene=False):
    if not obj_path.lower().endswith(".obj"):
        cmds.error("Expected an .obj file, got: {}".format(obj_path))
        return

    if clearScene:
        # Keep default cameras
        default_cams = {"persp", "top", "front", "side"}
        transforms = cmds.ls(type="transform") or []
        to_delete = [t for t in transforms if t not in default_cams]
        if to_delete:
            cmds.delete(to_delete)

    try:
        cmds.file(
            obj_path,
            i=True,
            type="OBJ",
            ignoreVersion=True,
            ra=True,
            mergeNamespacesOnClash=False,
            namespace=":",
            options="mo=1",  # "mo=1" = "merge objects" option in OBJ import
            pr=True
        )
        print("Imported OBJ:", obj_path)
    except Exception as e:
        cmds.error("Failed to import OBJ: {}".format(e))


# ccraft garment
obj_dir = "D:/ClothSim/ccraft_data/aux_data/garment_meshes/smpl/"
obj_name = "aaron_009__top.obj"

# tailornet garment
#obj_name = "t-shirt_male_000_000.obj"

# AMASS body
#obj_dir = "D:/ClothSim/AMASS/CMU/05/"
#obj_name = "body_frame000.obj"
spawn_obj(obj_dir+obj_name, clearScene=False)
