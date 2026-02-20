import pytest
import numpy as np
import trimesh
from cosmic_vine import CosmicVine, VineBlock  # Assuming your code is in cosmic_vine.py

def test_vine_block_mesh_creation():
    """Ensure a VineBlock creates a valid mesh with the correct size."""
    size = 1.0
    transform = np.eye(4)
    block = VineBlock(size=size, transform=transform, index=0)
    
    assert isinstance(block.mesh, trimesh.Trimesh)
    # Check that the bounds match a 1.0 size box centered at origin
    assert np.allclose(block.mesh.extents, [size, size, size])

def test_vine_growth_count():
    """Ensure the vine grows the exact number of blocks requested."""
    iterations = 10
    vine = CosmicVine(block_size=1.0)
    vine.grow(iterations=iterations)
    
    assert len(vine.blocks) == iterations
    combined = vine.get_combined_mesh()
    # Each box has 12 triangular faces
    assert len(combined.faces) == iterations * 12

def test_transformation_accumulation():
    """Verify that the second block is translated correctly from the first."""
    vine = CosmicVine(block_size=1.0)
    # Grow with 0 rotation to make math predictable
    vine.grow(iterations=2, max_rotation_rad=0)
    
    first_block_pos = vine.blocks[0].transform[:3, 3]
    second_block_pos = vine.blocks[1].transform[:3, 3]
    
    # The center of the second block should be exactly 1 unit away on Z
    dist = np.linalg.norm(second_block_pos - first_block_pos)
    assert np.isclose(dist, 1.0)
    assert np.allclose(second_block_pos, [0, 0, 1.0])

def test_flush_connectivity():
    """
    Crucial Test: Ensure blocks are flush. 
    The distance between centers must equal block_size even with rotation.
    """
    size = 1.0
    vine = CosmicVine(block_size=size)
    vine.grow(iterations=5, max_rotation_rad=np.pi/4)
    
    for i in range(len(vine.blocks) - 1):
        pos_a = vine.blocks[i].transform[:3, 3]
        pos_b = vine.blocks[i+1].transform[:3, 3]
        distance = np.linalg.norm(pos_b - pos_a)
        
        # In this specific architecture, the transform is applied to the center.
        # So distance between consecutive centers must always be exactly block_size.
        assert np.isclose(distance, size), f"Gap detected between block {i} and {i+1}"

def test_mesh_concatenation_validity():
    """Check if the final combined mesh is 'watertight' (optional but good for formX)."""
    vine = CosmicVine(block_size=1.0)
    vine.grow(iterations=3)
    mesh = vine.get_combined_mesh()
    
    assert not mesh.is_empty
    assert mesh.is_volume  # A vine of cubes should technically be a valid volume