import numpy as np
import trimesh
import random
from typing import List, Optional

class VineBlock:
    """Represents a single segment of the vine with its own transformation."""
    def __init__(self, size: float, transform: np.ndarray, index: int):
        self.size = size
        self.transform = transform
        self.index = index
        self.mesh = self._generate_mesh()

    def _generate_mesh(self) -> trimesh.Trimesh:
        mesh = trimesh.creation.box(extents=[self.size] * 3)
        mesh.apply_transform(self.transform)
        
        # Apply formX style aesthetic: subtle color gradient
        color_val = max(50, 255 - (self.index * 8))
        mesh.visual.face_colors = [100, color_val, 180, 255]
        return mesh

class CosmicVine:
    """Manager class to handle the growth logic and coordinate systems."""
    def __init__(self, block_size: float = 1.0):
        self.block_size = block_size
        self.blocks: List[VineBlock] = []
        self._current_transform = np.eye(4)

    def grow(self, iterations: int, max_rotation_rad: float = np.pi/4):
        """Core procedural growth algorithm."""
        for i in range(iterations):
            # Create and store the current segment
            new_block = VineBlock(self.block_size, self._current_transform.copy(), i)
            self.blocks.append(new_block)

            # Calculate the 'Jump' to the next segment's origin
            # Step 1: Translate to the face of the next block
            translation = trimesh.transformations.translation_matrix([0, 0, self.block_size])
            
            # Step 2: Generate random orientation
            angles = [random.uniform(-max_rotation_rad, max_rotation_rad) for _ in range(3)]
            rotation = trimesh.transformations.euler_matrix(*angles)
            
            # Step 3: Accumulate the transform (Local to Global)
            step_transform = np.dot(translation, rotation)
            self._current_transform = np.dot(self._current_transform, step_transform)

    def get_combined_mesh(self) -> trimesh.Trimesh:
        """Merges all block entities into a single exportable mesh."""
        return trimesh.util.concatenate([b.mesh for b in self.blocks])

class VineExporter:
    """Utility class to handle IO operations."""
    @staticmethod
    def save(vine: CosmicVine, filename: str):
        mesh = vine.get_combined_mesh()
        mesh.export(filename)
        print(f"Asset successfully baked to {filename}")

    @staticmethod
    def preview(vine: CosmicVine):
        mesh = vine.get_combined_mesh()
        mesh.show()

# --- Entry Point ---
if __name__ == "__main__":
    # Initialize the engine
    vine = CosmicVine(block_size=1.0)
    
    # Execute growth logic
    vine.grow(iterations=100, max_rotation_rad=np.pi/4)
    
    # Export and Preview
    VineExporter.save(vine, 'formx_cosmic_vine.obj')
    VineExporter.preview(vine)