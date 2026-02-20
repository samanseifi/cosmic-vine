"""
test_all.py — Unit tests for the Cosmic Vine procedural geometry system.

Run with:  pytest test_all.py -v
"""
import os
import tempfile

import numpy as np
import pytest
import trimesh

from cosmic_vine import CosmicVine, VineBlock, VineExporter


# ── VineBlock ──────────────────────────────────────────────────────────────────

class TestVineBlock:

    def test_creates_valid_trimesh(self):
        """VineBlock must produce a non-empty trimesh.Trimesh instance."""
        block = VineBlock(size=1.0, transform=np.eye(4), index=0, total=10)
        assert isinstance(block.mesh, trimesh.Trimesh)
        assert not block.mesh.is_empty

    def test_mesh_extents_match_size_at_identity(self):
        """
        With an identity transform the AABB extents must equal the requested
        size on all three axes.
        """
        for size in [0.5, 1.0, 3.0]:
            block = VineBlock(size=size, transform=np.eye(4), index=0, total=1)
            assert np.allclose(block.mesh.extents, [size, size, size])

    def test_mesh_has_twelve_faces(self):
        """A box is tessellated into exactly 12 triangular faces (2 per side × 6 sides)."""
        block = VineBlock(size=1.0, transform=np.eye(4), index=0, total=1)
        assert len(block.mesh.faces) == 12

    def test_mesh_has_eight_vertices(self):
        """A box has exactly 8 unique corner vertices."""
        block = VineBlock(size=1.0, transform=np.eye(4), index=0, total=1)
        assert len(block.mesh.vertices) == 8

    def test_transform_is_applied(self):
        """Applying a translation transform must shift the mesh centroid accordingly."""
        offset = np.array([3.0, -2.0, 7.0])
        transform = np.eye(4)
        transform[:3, 3] = offset
        block = VineBlock(size=1.0, transform=transform, index=0, total=1)
        assert np.allclose(block.mesh.centroid, offset, atol=1e-6)

    # ── Color gradient ────────────────────────────────────────────────────────

    def test_all_face_colors_are_valid_rgba(self):
        """Every RGBA channel must be in the 0-255 range for every block position."""
        total = 10
        for idx in range(total):
            block = VineBlock(size=1.0, transform=np.eye(4), index=idx, total=total)
            for channel in block.mesh.visual.face_colors[0]:
                assert 0 <= channel <= 255, (
                    f"Color channel out of range at index={idx}: {block.mesh.visual.face_colors[0]}"
                )

    def test_color_gradient_at_first_block(self):
        """First block (t = 0) must use the root values of the gradient formula."""
        block = VineBlock(size=1.0, transform=np.eye(4), index=0, total=10)
        r, g, b, a = block.mesh.visual.face_colors[0]
        assert r == 80,  f"Expected r=80, got {r}"
        assert g == 180, f"Expected g=180, got {g}"
        assert b == 220, f"Expected b=220, got {b}"
        assert a == 255

    def test_color_gradient_at_last_block(self):
        """Last block (t → 1) must use the tip values of the gradient formula."""
        total = 10
        idx   = total - 1
        t     = idx / total
        block = VineBlock(size=1.0, transform=np.eye(4), index=idx, total=total)
        r, g, b, a = block.mesh.visual.face_colors[0]
        assert r == int(80  + t * 60),  f"R mismatch: {r}"
        assert g == int(180 - t * 100), f"G mismatch: {g}"
        assert b == int(220 - t * 60),  f"B mismatch: {b}"
        assert a == 255

    def test_color_gradient_is_monotonic_in_r_and_not_flat(self):
        """
        The red channel should increase across blocks (gradient is progressive),
        confirming the gradient is not stuck at a single value.
        """
        total = 20
        reds = [
            VineBlock(size=1.0, transform=np.eye(4), index=i, total=total)
            .mesh.visual.face_colors[0][0]
            for i in range(total)
        ]
        assert reds[0] < reds[-1], "Red channel should increase from root to tip"


# ── CosmicVine ─────────────────────────────────────────────────────────────────

class TestCosmicVine:

    def _make_vine(self, n=10, rotation=np.pi / 4, size=1.0):
        vine = CosmicVine(block_size=size)
        vine.grow(iterations=n, max_rotation_rad=rotation)
        return vine

    # ── Block count ───────────────────────────────────────────────────────────

    def test_grow_produces_exact_block_count(self):
        """grow(n) must append exactly n blocks to the vine."""
        for n in [1, 10, 50]:
            vine = self._make_vine(n)
            assert len(vine.blocks) == n, f"Expected {n} blocks, got {len(vine.blocks)}"

    def test_single_block_edge_case(self):
        """A vine of one block is valid and should not raise."""
        vine = CosmicVine(block_size=1.0)
        vine.grow(iterations=1)
        assert len(vine.blocks) == 1

    # ── Initial transform ─────────────────────────────────────────────────────

    def test_first_block_is_at_world_origin(self):
        """Block 0 must always start at the world origin with identity orientation."""
        vine = self._make_vine()
        transform = vine.blocks[0].transform
        assert np.allclose(transform, np.eye(4), atol=1e-10)

    def test_zero_rotation_second_block_along_z(self):
        """
        With max_rotation_rad=0 the vine grows in a straight line along +Z.
        Block 1's center must be exactly one block_size unit ahead along Z.
        """
        vine = CosmicVine(block_size=1.0)
        vine.grow(iterations=2, max_rotation_rad=0)
        pos = vine.blocks[1].transform[:3, 3]
        assert np.allclose(pos, [0.0, 0.0, 1.0], atol=1e-10)

    def test_zero_rotation_all_orientations_are_identity(self):
        """With no rotation every block orientation must remain the identity matrix."""
        vine = CosmicVine(block_size=1.0)
        vine.grow(iterations=8, max_rotation_rad=0)
        for i, block in enumerate(vine.blocks):
            rot = block.transform[:3, :3]
            assert np.allclose(rot, np.eye(3), atol=1e-10), (
                f"Block {i} orientation is not identity:\n{rot}"
            )

    # ── Flush face connectivity (the core geometric constraint) ───────────────

    def test_flush_face_connectivity_standard(self):
        """
        The 'perfectly flush' requirement: the +Z face center of block N must
        land exactly on the -Z face center of block N+1 in world space.

        Face centers are computed via homogeneous transform:
            +Z face of block N   = transform_N   @ [0, 0, +b/2, 1]
            -Z face of block N+1 = transform_N+1 @ [0, 0, -b/2, 1]

        These must agree to floating-point precision for all consecutive pairs.
        This test would FAIL with the naive step  T_z(b) @ R  and only passes
        with the correct  T_z(b/2) @ R @ T_z(b/2)  pivot formulation.
        """
        vine = self._make_vine(n=20, rotation=np.pi / 4)
        h = vine.block_size / 2
        for i in range(len(vine.blocks) - 1):
            t_a = vine.blocks[i].transform
            t_b = vine.blocks[i + 1].transform
            face_out = (t_a @ np.array([0.0, 0.0,  h, 1.0]))[:3]
            face_in  = (t_b @ np.array([0.0, 0.0, -h, 1.0]))[:3]
            assert np.allclose(face_out, face_in, atol=1e-10), (
                f"Flush violation between blocks {i} and {i+1}:\n"
                f"  +Z face of block {i}   = {face_out}\n"
                f"  -Z face of block {i+1} = {face_in}"
            )

    def test_flush_at_maximum_rotation(self):
        """Flush constraint must hold even at the maximum 90° rotation."""
        vine = self._make_vine(n=15, rotation=np.pi / 2)
        h = vine.block_size / 2
        for i in range(len(vine.blocks) - 1):
            face_out = (vine.blocks[i].transform     @ np.array([0.0, 0.0,  h, 1.0]))[:3]
            face_in  = (vine.blocks[i + 1].transform @ np.array([0.0, 0.0, -h, 1.0]))[:3]
            assert np.allclose(face_out, face_in, atol=1e-10), (
                f"Flush violation at π/2 rotation between blocks {i} and {i+1}"
            )

    def test_flush_with_non_unit_block_size(self):
        """Flush constraint must hold regardless of block_size."""
        for size in [0.25, 2.0, 5.0]:
            vine = CosmicVine(block_size=size)
            vine.grow(iterations=10, max_rotation_rad=np.pi / 3)
            h = size / 2
            for i in range(len(vine.blocks) - 1):
                face_out = (vine.blocks[i].transform     @ np.array([0.0, 0.0,  h, 1.0]))[:3]
                face_in  = (vine.blocks[i + 1].transform @ np.array([0.0, 0.0, -h, 1.0]))[:3]
                assert np.allclose(face_out, face_in, atol=1e-10), (
                    f"Flush violation for block_size={size} between blocks {i} and {i+1}"
                )

    def test_flush_zero_rotation_trivially_holds(self):
        """Sanity check: flush also holds for the degenerate zero-rotation case."""
        vine = CosmicVine(block_size=1.0)
        vine.grow(iterations=5, max_rotation_rad=0)
        h = vine.block_size / 2
        for i in range(len(vine.blocks) - 1):
            face_out = (vine.blocks[i].transform     @ np.array([0.0, 0.0,  h, 1.0]))[:3]
            face_in  = (vine.blocks[i + 1].transform @ np.array([0.0, 0.0, -h, 1.0]))[:3]
            assert np.allclose(face_out, face_in, atol=1e-10)

    # ── Step transform math ───────────────────────────────────────────────────

    def test_step_transform_pivot_is_at_face_not_center(self):
        """
        Directly verify the step decomposition T(b/2)@R@T(b/2) by checking
        that the connecting face point [0,0,b/2] maps to itself under the step.

        If the wrong formulation T(b)@R were used, this face point would move.
        """
        import trimesh.transformations as tf
        block_size = 1.0
        h = block_size / 2
        # Use a large rotation to make the difference obvious
        angles = [np.pi / 3, np.pi / 4, np.pi / 6]
        rotation  = tf.euler_matrix(*angles)
        half_step = tf.translation_matrix([0, 0, h])
        step      = half_step @ rotation @ half_step

        face_point = np.array([0.0, 0.0, h, 1.0])
        # The -Z face of the next block in the NEXT block's local frame is [0,0,-h,1].
        # After applying step to [0,0,-h,1] we should get back to [0,0,h,1] (the shared face).
        mapped = step @ np.array([0.0, 0.0, -h, 1.0])
        assert np.allclose(mapped[:3], face_point[:3], atol=1e-10), (
            f"Step transform does not preserve face point:\n"
            f"  expected {face_point[:3]}, got {mapped[:3]}"
        )

    # ── Combined mesh ─────────────────────────────────────────────────────────

    def test_combined_mesh_face_count(self):
        """Concatenated mesh must have exactly n × 12 faces (12 per box)."""
        n = 7
        vine = self._make_vine(n=n)
        combined = vine.get_combined_mesh()
        assert len(combined.faces) == n * 12

    def test_combined_mesh_is_not_empty(self):
        assert not self._make_vine(n=5).get_combined_mesh().is_empty

    def test_combined_mesh_vertex_count(self):
        """Concatenated mesh must have exactly n × 8 vertices (8 per box)."""
        n = 6
        vine = self._make_vine(n=n)
        assert len(vine.get_combined_mesh().vertices) == n * 8


# ── VineExporter ───────────────────────────────────────────────────────────────

class TestVineExporter:

    def _make_vine(self, n=5):
        vine = CosmicVine(block_size=1.0)
        vine.grow(iterations=n)
        return vine

    def test_save_creates_file_on_disk(self):
        """save() must write a non-empty file at the specified path."""
        vine = self._make_vine()
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False) as f:
            path = f.name
        try:
            VineExporter.save(vine, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.remove(path)

    def test_save_produces_valid_obj_text(self):
        """The exported .OBJ content must contain vertex ('v') and face ('f') lines."""
        vine = self._make_vine()
        with tempfile.NamedTemporaryFile(suffix='.obj', delete=False, mode='r') as f:
            path = f.name
        try:
            VineExporter.save(vine, path)
            content = open(path).read()
            assert 'v '  in content, "OBJ is missing vertex lines"
            assert 'f '  in content, "OBJ is missing face lines"
        finally:
            os.remove(path)

    def test_screenshot_creates_png_file(self):
        """screenshot() must write a non-empty PNG at the specified path."""
        vine = self._make_vine()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            VineExporter.screenshot(vine, path)
            assert os.path.exists(path)
            assert os.path.getsize(path) > 0
        finally:
            os.remove(path)

    def test_screenshot_file_is_valid_png(self):
        """The screenshot output must begin with the PNG magic bytes (\\x89PNG)."""
        vine = self._make_vine()
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            path = f.name
        try:
            VineExporter.screenshot(vine, path)
            with open(path, 'rb') as f:
                header = f.read(4)
            assert header == b'\x89PNG', f"File does not start with PNG header: {header}"
        finally:
            os.remove(path)
