import maya.cmds as cmds

sel = cmds.ls(selection=True)
nodes = []
for node in sel:
    node_tuple = node[12:-1]
    if ":" in node_tuple:
        i = int(node_tuple.split(":")[0])
        j = int(node_tuple.split(":")[1])
        nodes += [x for x in range(i, j+1)]
    else:
        nodes.append(int(node_tuple))
print("Selected nodes:", nodes)

