import hou
import numpy as np
from fkskeleton import FKSkeleton

"""
Collection of Utils functions to work with Houdini
"""

def toKinefxGeo(fk: FKSkeleton) -> hou.Geometry:
    """
    Convert a fk skeleton into a Kinefx ready geometry
    """
    geo = hou.Geometry()

    geo.addAttrib(hou.attribType.Point,'name','')
    names = fk.allJointNames()
    for i in range(fk.jointCount()):
        name = names[i]
        pt = geo.createPoint()
        pt.setAttribValue("name", name)

    parent_hierarchy = fk.parentHierarchy().tolist()
    for childId, parentId in enumerate(parent_hierarchy):
        if parentId < 0:
            continue
        childPt = geo.point(childId)
        parentPt = geo.point(parentId)
        prim = geo.createPolygon(False)
        prim.addVertex(parentPt)
        prim.addVertex(childPt)

    return geo

def toAgentRig(fk: FKSkeleton, rigname="defaultRig") -> hou.AgentRig:
    """"
    Convert a FK Skeleton into an hou.AgentRig
    """
    hierarchy = [ children.tolist() for children in fk.transformHierarchy() ]
    return hou.AgentRig(rigname, fk.allJointNames(), hierarchy)

def fromAgentRig(rig: hou.AgentRig) -> FKSkeleton:
    """
    Convert a hou.AgentRig into a FKSkeleton
    """
    hierarchy = [ rig.parentIndex(jnt) for jnt in range(rig.transformCount()) ]
    joint_names = list(rig.transformNames())
    skel = FKSkeleton(hierarchy, joint_names)

    return skel