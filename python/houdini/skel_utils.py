import hou
import numpy as np
from fkskeleton import FKSkeleton

"""
Collection of Utils functions to work with Houdini
"""

def toKinefxGeo(fk: FKSkeleton) -> hou.Geometry:
    """
    Convert a fk skeleton into a Kinefx ready geometry. A KineFX geometry is a geometry such that:
    Each joint is represented by a Point. A Parent/Child relationship is represented as a Polyline Primitive
    such that vertex0:vertex1 is parent:child.
    Position and orientation of the joints are set from the rest transform of the FKSkeleton
    Args:
        fk: FKSkeleton the skeleton to convert
    Output:
        hou.Geometry: a read/write hou.Geometry containing the skeleton represented as a KineFX rig
    """
    geo = hou.Geometry()

    geo.addAttrib(hou.attribType.Point,'name','')
    geo.addAttrib(hou.attribType.Point,'transform',np.eye(3).flatten())
    geo.addAttrib(hou.attribType.Point,'localtransform',np.eye(4).flatten())
    names = fk.allJointNames()
    xforms = fk.worldRestPose()
    localxforms = fk.localRestPose().transpose((0,1,3,2))
    
    for i in range(fk.jointCount()):
        name = names[i]
        pt = geo.createPoint()
        pt.setAttribValue("name", name)

    t = xforms[:,:,:3,3].flatten()
    xforms = xforms.transpose((0,1,3,2)) #Houdini has inverted representation
    transform = xforms[:,:,:3,:3].flatten() 

    geo.setPointFloatAttribValues('P',t)
    geo.setPointFloatAttribValues('transform', transform)
    geo.setPointFloatAttribValues('localtransform', localxforms.flatten())

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

def fromKinefxGeo(geo: hou.Geometry) -> FKSkeleton:
    """
    Convert a Kinefx skeleton into a FKSkeleton
    Args:
        geo: hou.Geometry Geometry representing a Kinefx rig
    Output:
        FKSkeleton: equivalent of the Kinefx geo
    """

    # We rely on the agentfromrig verb to convert to agent
    agentgeo = hou.Geometry()
    convert_verb = hou.sopNodeTypeCategory().nodeVerb("kinefx::agentfromrigcore")
    convert_verb.execute(agentgeo, [geo])
    
    agent = agentgeo.prim(0)
    return fromAgentRig(agent.rig())


def toAgentRig(fk: FKSkeleton, rigname="defaultRig") -> hou.AgentRig:
    """"
    Convert a FK Skeleton into an hou.AgentRig
    Args:
        fk: FKSkeleton the skeleton to convert to AgentRig
        rigname: str the name provided to the hou.AgentRig
    Output:
        hou.AgentRig equivalent to fk
    """
    hierarchy = [ children.tolist() for children in fk.transformHierarchy() ]
    return hou.AgentRig(rigname, fk.allJointNames(), hierarchy)

def fromAgentRig(rig: hou.AgentRig) -> FKSkeleton:
    """
    Convert a hou.AgentRig into a FKSkeleton
    Args:
        rig: hou.AgentRig a Houdini Agent Rig
    Output:
        FKSkeleton: the corresponding FK Skeleton. Note that it has less information than the AgentRig
    """
    hierarchy = [ rig.parentIndex(jnt) for jnt in range(rig.transformCount()) ]
    joint_names = list(rig.transformNames())
    rest_transforms = np.array([rig.restLocalTransform(jnt).asTuple() for jnt in range(rig.transformCount())])

    rest_transforms = rest_transforms.reshape((1,rig.transformCount() ,4, 4)).transpose((0,1,3,2))
    skel = FKSkeleton(hierarchy, joint_names, rest_transforms)

    return skel
