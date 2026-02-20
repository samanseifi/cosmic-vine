import numpy as np
import trimesh
import random
import matplotlib.pyplot as plt
from matplotlib.colors import hsv_to_rgb

class CosmicStrand:
    def __init__(self, iterations=100, block_size=0.01, max_rot=np.pi/4):
        self.block_size = block_size
        self.iterations = iterations
        self.max_rot = max_rot
        self.points = []
        self.curvatures = [] # Measured in radians (angle delta)
        self.transforms = []

    def grow(self, start_transform=None):
        current_tf = start_transform if start_transform is not None else np.eye(4)
        
        for i in range(self.iterations):
            # Record current origin
            self.points.append(current_tf[:3, 3])
            self.transforms.append(current_tf.copy())
            
            # 1. Step forward
            translation = np.eye(4)
            translation[2, 3] = self.block_size
            
            # 2. Random Rotate
            angles = [random.uniform(-self.max_rot, self.max_rot) for _ in range(3)]
            rotation = trimesh.transformations.euler_matrix(*angles)
            
            # 3. Update
            current_tf = current_tf @ translation @ rotation

        self._calculate_discrete_curvature()

    def _calculate_discrete_curvature(self):
        """Calculates the angle between segment i and segment i+1."""
        self.curvatures = [0.0] # First point has no curvature
        
        for i in range(1, len(self.points) - 1):
            # Vector A: segment before point i
            v_a = self.points[i] - self.points[i-1]
            # Vector B: segment after point i
            v_b = self.points[i+1] - self.points[i]
            
            # Unit vectors
            v_a_n = v_a / np.linalg.norm(v_a)
            v_b_n = v_b / np.linalg.norm(v_b)
            
            # Angle via dot product
            dot = np.clip(np.dot(v_a_n, v_b_n), -1.0, 1.0)
            angle = np.arccos(dot)
            self.curvatures.append(angle)
            
        self.curvatures.append(0.0) # Last point

    def get_colored_path(self):
        """Creates a Path3D by defining each segment as a unique entity for coloring."""
        points = np.array(self.points)
        
        # We define each segment as a separate Line entity
        entities = []
        for i in range(len(points) - 1):
            # Entity refers to the index of the points it connects
            entities.append(trimesh.path.entities.Line([i, i + 1]))
            
        # Create the Path3D with explicit entities
        path = trimesh.path.Path3D(entities=entities, vertices=points)
        
        # Now we can map colors per-entity (per-segment)
        segment_colors = []
        max_c = self.max_rot
        
        for i in range(len(entities)):
            # Hue: 0.6 (Blue) for straight, 0.0 (Red) for sharp
            # We use the curvature calculated for the starting point of the segment
            hue = np.clip(0.6 * (1.0 - (self.curvatures[i] / max_c)), 0, 0.6)
            rgb = hsv_to_rgb([hue, 1.0, 1.0]) * 255
            segment_colors.append(rgb.tolist() + [255]) # Add Alpha

        path.colors = np.array(segment_colors, dtype=np.uint8)
        return path

class StrandForest:
    def __init__(self, num_strands=100):
        self.scene = trimesh.Scene()
        for _ in range(num_strands):
            # Random starting positions
            pos = np.random.uniform(-20, 20, 3)
            start_tf = trimesh.transformations.translation_matrix(pos)
            
            strand = CosmicStrand(iterations=1000, max_rot=np.pi/6)
            strand.grow(start_transform=start_tf)
            self.scene.add_geometry(strand.get_colored_path())

    def show(self):
        self.scene.show()

if __name__ == "__main__":
    forest = StrandForest(num_strands=1)
    forest.show()