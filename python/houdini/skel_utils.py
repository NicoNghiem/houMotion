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
    geo.addAttrib(hou.attribType.Point,'transform',np.eye(3).flatten())
    names = fk.allJointNames()
    xforms = fk.worldRestPose()
    
    for i in range(fk.jointCount()):
        name = names[i]
        pt = geo.createPoint()
        pt.setAttribValue("name", name)

    t = xforms[:,:,:3,3].flatten()
    xforms = xforms.transpose((0,1,3,2))
    transform = xforms[:,:,:3,:3].flatten() #Houdini has inverted representation

    geo.setPointFloatAttribValues('P',t)
    geo.setPointFloatAttribValues('transform', transform)

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
    rest_transforms = np.array([rig.restLocalTransform(jnt).asTuple() for jnt in range(rig.transformCount())])

    rest_transforms = rest_transforms.reshape((1,rig.transformCount() ,4, 4)).transpose((0,1,3,2))
    skel = FKSkeleton(hierarchy, joint_names, rest_transforms)

    return skel
