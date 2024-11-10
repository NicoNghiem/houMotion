# houMotion
A collection of code used to work with Motion in Houdini

## Usage

The FKSkeleton class can be used as a standalone but we provide utils in skelutils to quickly create FKSkeleton to be used inside Houdini. We also provide ways to convert FKSkeleton to either Kinefx geometry or a hou.AgentRig so that it is possible to store things as a .bgeo file

### Convert Agent to FKSkeleton

We assume the Agent is the first Prim of the first input geometry in a Python SOP

`import hou
from fkskeleton import FKSkeleton
import houdini.skel_utils as skelutils

agent = hou.pwd().geometry().prim(0)

skel = skelutils.fromAgentRig(agent.rig())`

### Convert FKSkeleton into a Kinefx skeleton

This code is written in a Python SOP to contain the geometry

`import hou
from fkskeleton import FKSkeleton
import houdini.skel_utils as skelutils

'''Create a chain of 26 joints making a line'''
parents = np.arange(-1,25).tolist()
xforms = np.zeros((1,26,4,4))
xforms[:,:] = np.eye(4)
xforms[:,:,0,3] = 0.1

skel = FKSkeleton(parents, rest_transforms=xforms)

hou.pwd().geometry().merge(skelutils.toKinefxGeo(skel))`