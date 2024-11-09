import numpy as np

class fkskeleton:
    """
    Represents a FK skeleton
    A FK Skeleton is a hierarchy where:
    -> each joint has at most one parent (zero parent is a Root)
    -> each joint can have have multiple children

    Attributes:
    
    Private attributes:

    _parent_hierarchy: the Hierarchy represented as an array where array[i] = parent of i
                        np.array of size (jointCount,) of int
                        It is the main representation of the Topology and is used to create the skeleton
    _matrix_hierarchy (*): the Hierarchy represented as a matrix where array[i][j]=1  iff i is the parent of j
                        np.array of size (jointCount,jointCount) of int
    _depth_hierarchy (*): the Hierarchy represented as an array of size (maxDepth, jointCount)
                        its main purpose is to let us know about operation order
                        np.array of size (maxDepth, jointCount) of int
    _transform_hierarchy (*): the Hierarchy represented as an array where array[i] = children of i
                        np.array of size (jointCount,) of np.array
    _joint_names (optional): the names of the joints. will fill any non given names by "joint_idx"

    (*) those attributes are None by default but are cached the first time they are computed
    """

    _parent_hierarchy = None
    _matrix_hierarchy = None
    _depth_hierarchy = None
    _transform_hierarchy = None
    _joint_names = None

    def __init__(self, parent_hierarchy=[], joint_names=[],check=True):
        """
        Initialize given an hierarchy and joint names
        """
        if isinstance(parent_hierarchy, list):
            self._parent_hierarchy = np.array(parent_hierarchy).astype(int)
        else:
            self._parent_hierarchy = parent_hierarchy.astype(int)

        if isinstance(joint_names, list):
            self._joint_names = np.array(joint_names)
        else:
            self._joint_names = joint_names

        self._validate()

    def _validate(self):
        """
        Check if the given skeleton is valid.
        We check if:
            -> all parents values exist
            -> there is no cycle
        """
        #check if all parents exist
        detect = np.where(self.parentHierarchy()>=self.jointCount())[0]
        if len(detect)>=1:
            raise ValueError("Detected Out of Bound Parent index")
        #check that we have at least one root
        if self.jointCount()==0:
            raise ValueError("No root detected")
        #check there are no cycles
        #one easy way to do this is to compute the depth hierarchy as it fails if there is a cycle
        self._buildDepthHierarchy()

        #fill the names with "joint_idx" if missing names
        name_count = self._joint_names.shape[0]
        if name_count>=self.jointCount():
            self._joint_names == self._joint_names[:self.jointCount()]
        elif name_count < (self.jointCount()-1):
            idx = np.arange(name_count, self.jointCount())
            prefix = np.full(idx.shape, "joint_", dtype='object')
            idx = np.add(prefix, idx.astype('str'))
            if name_count!=0:
                self._joint_names = np.r_[self._joint_names, idx]
            else:
                self._joint_names = idx
        return

    def _buildMatrixHierarchy(self):
        """
        Builds the Matrix hierarchy:
        Matrix of size (Njnts, Njnts) where:
        M[i,j] = 1 if i is j's parent, 0 otherwise
        """
        M = np.zeros((self.jointCount(), self.jointCount()))
        idx = np.arange(self.jointCount())
        idx = np.delete(idx, self.roots()) #remove roots
        M[idx, self.getParent(idx)] = 1
        self.matrixHierarchy = M
        return
    
    def _buildTransformHierarchy(self):
        """
        Warning: this call is relatively slow compared to the other ones
        Builds the Transform Hierarchy:
        xformHierarchy[jnt] = children of jnt
        """
        parentHierarchy = self.parentHierarchy()
        self._transform_hierarchy = np.array([ np.where(parentHierarchy==idx)[0].astype(int) for idx in range(self.jointCount()) ] , dtype=object)
        return
    
    def _buildDepthHierarchy(self):
        """
        Builds the depth hierarchy:
        Similar to parent hierarchy but concatenated as a Matrix. It allows us to store the depth information without having to traverse the rig
        np.array of size (max_depth, Njoints) where each entry is either -1 or the index of your parent
        """
        depthHierarchy = -np.ones((1, self.jointCount())).astype(int)
        c_jnts = self.roots()
        
        iter = 0
        while len(c_jnts)>0 :
            #breaking clause in case there is a cycle
            if iter > self.jointCount():
                raise ValueError("Cycle detected")
                break
                
            parents = c_jnts
            children = self.getChildren(parents)
            children  = np.concatenate(children)
            
            if len(children)>0:
                row = -np.ones((1, self.jointCount())).astype(int)
                row[:, children] = self.getParent(children)
                depthHierarchy = np.r_[depthHierarchy, row]
            
            c_jnts = children
            iter+=1
        self._depth_hierarchy = depthHierarchy[1:,:]
        self._max_depth = iter
        return
    
    #Public Methods

    # Hierarchy methods

    def parentHierarchy(self) -> np.array:
        """
        Returns the Parent Hierarchy of the skeleton
        parent_hierarchy[i] = parent of joint i
        """
        return self._parent_hierarchy
    
    def depthHierarchy(self) -> np.array:
        """
        Returns the depth hierarchy
        Similar to parent hierarchy but concatenated as a Matrix. 
        np.array of size (max_depth, Njoints) where each entry is either -1 or the index of your parent
        """
        if self._depth_hierarchy is None:
            self._buildDepthHierarchy()
        return self._depth_hierarchy
    
    def transformHierarchy(self) -> np.array:
        """
        Returns the transform hierarchy
        transform_hierarchy[i] = np.array containing the children of i
        """
        if self._transform_hierarchy is None:
            self._buildTransformHierarchy()
        return self._transform_hierarchy
    
    def maxDepth(self) -> np.array:
        if self._depth_hierarchy is None:
            self._buildDepthHierarchy()
        return self._max_depth

    def getParent(self, children: np.array) -> np.array:
        """
        Returns an array containing the parents of the given joints
        Returns -1 if the joint is a root and has no parents
        """
        return self.parentHierarchy()[children]
    
    def getChildren(self, parents: np.array) -> np.array:
        """
        Returns an array containing the parents of the given joints (by id)
        Returns empty array if joint has no children 
        """
        return self.transformHierarchy()[parents]

    def roots(self) -> np.array:
        """
        Returns the roots of the fkskeleton (aka the joints that do not have a parent)
        """
        return np.where(self.parentHierarchy()==-1)[0].astype(int)
    
    def jointCount(self) -> int:
        """
        Returns the Number of joint of the fkskeleton
        """
        return self.parentHierarchy().shape[0]

    # Name methods

    def allJointNames(self) -> np.array:
        """
        Returns all the joint names
        """
        return self._joint_names

    def jointName(self, ids: np.array) -> np.array:
        """
        Returns the names of the queried joints
        """
        return self.allJointNames()[ids]
    
    def jointId(self, names: np.array) -> np.array:
        """
        Returns the ids of the given joint names or -1 if it does not exist
        """
        all_names = self.allJointNames()
        sorter = np.argsort(all_names)

        idx = np.searchsorted(all_names, names, sorter=sorter)
        # this would return 0 for names that do not exist so we need to identify them
        test_idx = np.where(idx==0)[0]
        wrong_idx = np.where(names[test_idx]!=all_names[0])[0]
        idx[test_idx[wrong_idx]]=-1

        return idx

    # Rig evaluation methods

    def localFromWorld(self, worldXforms: np.array) -> np.array:
        """
        From a batch of poses in World space, compute the corresponding pose in Local Space
        worldXforms: np.array of size (:, jointCount, 4, 4)

        output: np.array of size (:, jointCount, 4, 4)
        """
        local = worldXforms.copy()
        childIdx = np.arange(self.jointCount()).delete(self.roots())
        parentIdx = self.getParent(childIdx)
        local[:,childIdx,:,:] = worldXforms[:,childIdx,:,:] @ np.linalg.inv(world[:,parentIdx,:,:])
        return local
    
    def worldFromLocal(self, localXforms: np.array) -> np.array:
        """
        From a batch of poses in Local Space, compute the corresponding pose in World Space
        localXforms: np.array of size (:, jointCount, 4, 4)
        
        output: np.array of size (:, jointCount, 4, 4)
        """
        world = localXforms.copy()
        depthHierarchy = self.depthHierarchy()
        for row in depthHierarchy:
                childIdx = np.where(row>=0)[0].astype(int)
                parentIdx = row[childIdx]
                world[:,childIdx,:,:] = local[:,childIdx,:,:] @ world[:,parentIdx,:,:]
        return world
    
    