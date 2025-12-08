import maya.cmds as cmds

sel = cmds.ls(selection=True)
verts = []
for vert in sel:
    vert = vert[12:-1]
    if ":" in vert:
        i = int(vert.split(":")[0])
        j = int(vert.split(":")[1])
        verts += [x for x in range(i, j+1)]
    else:
        verts.append(int(vert))
print("Selected verts:", verts)

