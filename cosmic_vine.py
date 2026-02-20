import numpy as np
import trimesh
import random
from typing import List, Optional
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


class VineBlock:
    """
    Represents a single cube segment of the vine.

    Each block owns its 4x4 world-space transform (position + orientation)
    and generates its own colored trimesh geometry on construction.
    """

    def __init__(self, size: float, transform: np.ndarray, index: int, total: int):
        """
        Parameters
        ----------
        size      : side length of the cube (same in all three axes)
        transform : 4x4 homogeneous matrix that places this block in world space
        index     : position of this block in the growth sequence (0 = first)
        total     : total number of blocks in the vine, used to normalize the color gradient
        """
        self.size = size
        self.transform = transform
        self.index = index
        self.total = total
        self.mesh = self._generate_mesh()

    def _generate_mesh(self) -> trimesh.Trimesh:
        """
        Builds a unit cube, moves it into world space, and assigns a color.

        The cube is created at the local origin by trimesh, then transformed
        into its final world-space position/orientation via apply_transform.

        Color shifts from cool blue (first block) to warm teal (last block)
        using a normalized parameter t = index / total, so the gradient
        spans the full vine regardless of how many blocks are grown.
        """
        # Create a cube centered at the local origin
        mesh = trimesh.creation.box(extents=[self.size] * 3)

        # Move the cube from local origin to its world-space position and orientation
        mesh.apply_transform(self.transform)

        # Normalized progress along the vine: 0.0 at start, 1.0 at end
        t = self.index / max(self.total, 1)

        # RGB gradient: blue-cyan at the root, shifting toward teal at the tip
        r = int(80 + t * 60)   # 80 → 140
        g = int(180 - t * 100) # 180 → 80
        b = int(220 - t * 60)  # 220 → 160
        mesh.visual.face_colors = [r, g, b, 255]
        return mesh


class CosmicVine:
    """
    Manages the procedural growth of the vine and the global coordinate chain.

    Growth works by maintaining a running 4x4 transform that represents the
    current "tip" of the vine. At each step the transform is advanced by one
    block-length and randomly rotated, accumulating in local space so that
    every new block inherits the orientation of the previous one.
    """

    def __init__(self, block_size: float = 1.0):
        """
        Parameters
        ----------
        block_size : side length of each cube and the center-to-center
                     distance between consecutive blocks
        """
        self.block_size = block_size
        self.blocks: List[VineBlock] = []

        # The running tip transform, initialized to the world origin
        self._current_transform = np.eye(4)

    def grow(self, iterations: int, max_rotation_rad: float = np.pi / 4):
        """
        Procedurally grows the vine one block at a time.

        At each iteration:
          1. A new VineBlock is placed at the current tip transform.
          2. A random Euler rotation (pitch / roll / yaw) is generated.
          3. The tip transform is advanced by one block-length using the
             decomposed step  T(b/2) @ R @ T(b/2), where b = block_size.

        The decomposed step is the key to the "perfectly flush" constraint.
        Naively doing T(b) @ R would pivot the rotation around the CENTER
        of the next block, which shifts its incoming face away from the
        current block's outgoing face — creating a gap.

        Instead, T(b/2) @ R @ T(b/2) decomposes the move as:
          - T(b/2) : advance to the connecting face in local Z
          - R       : rotate there (pivot is now at the shared face)
          - T(b/2) : advance to the center of the next block in its new local Z
        This guarantees the -Z face of block N+1 lands exactly on the
        +Z face of block N, with no gap or interpenetration.

        Parameters
        ----------
        iterations       : number of blocks to grow
        max_rotation_rad : maximum random rotation on each axis per step (radians)
        """
        for i in range(iterations):
            # Snapshot the current tip transform and create a block there
            new_block = VineBlock(self.block_size, self._current_transform.copy(), i, iterations)
            self.blocks.append(new_block)

            # Half-step translation along the current local Z axis
            half_step = trimesh.transformations.translation_matrix([0, 0, self.block_size / 2])

            # Random pitch / roll / yaw within the allowed range
            angles = [random.uniform(-max_rotation_rad, max_rotation_rad) for _ in range(3)]
            rotation = trimesh.transformations.euler_matrix(*angles)

            # Compose the flush step: face → rotate → center of next block
            step_transform = half_step @ rotation @ half_step

            # Accumulate in local space (right-multiply keeps everything relative
            # to the current block's own coordinate frame)
            self._current_transform = self._current_transform @ step_transform

    def get_combined_mesh(self) -> trimesh.Trimesh:
        """
        Merges every block's mesh into a single trimesh object.

        Useful for exporting the entire vine as one .OBJ file or for
        passing to the interactive viewer.
        """
        return trimesh.util.concatenate([b.mesh for b in self.blocks])


class VineExporter:
    """
    Handles all output operations: file export, screenshot, and live preview.
    """

    @staticmethod
    def save(vine: CosmicVine, filename: str):
        """
        Exports the full vine as a single .OBJ (or any format trimesh supports).

        All block meshes are first concatenated into one object so the output
        file contains a single mesh rather than one object per block.
        """
        mesh = vine.get_combined_mesh()
        mesh.export(filename)
        print(f"Asset successfully baked to {filename}")

    @staticmethod
    def screenshot(vine: CosmicVine, filename: str):
        """
        Renders a static PNG of the vine using matplotlib for the deliverable.

        Each block's triangulated faces are drawn as a Poly3DCollection on a
        dark background. Older blocks (closer to the root) are rendered with
        higher opacity so the vine reads clearly from root to tip.

        Axis limits are computed from the bounding box of all vertices and
        padded to equal span on all three axes, preventing distortion.
        """
        fig = plt.figure(figsize=(10, 10), facecolor='#0a0a0f')
        ax = fig.add_subplot(111, projection='3d', facecolor='#0a0a0f')

        for block in vine.blocks:
            mesh = block.mesh
            verts = mesh.vertices
            faces = mesh.faces

            # Normalize the stored RGBA color from 0-255 to 0.0-1.0 for matplotlib
            fc = mesh.visual.face_colors[0] / 255.0

            # Build a list of triangles (each triangle = list of 3 xyz vertices)
            polys = [[verts[j] for j in face] for face in faces]

            # Tip blocks are fully opaque; root blocks are slightly transparent,
            # giving a sense of depth without extra lighting setup
            t = block.index / max(len(vine.blocks), 1)
            alpha = 0.5 + 0.5 * (1 - t)

            collection = Poly3DCollection(polys, alpha=alpha, linewidths=0)
            collection.set_facecolor(fc[:3])
            ax.add_collection3d(collection)

        # Compute a square bounding box so no axis is stretched
        all_verts = np.vstack([b.mesh.vertices for b in vine.blocks])
        mins, maxs = all_verts.min(axis=0), all_verts.max(axis=0)
        mid = (mins + maxs) / 2
        span = (maxs - mins).max() / 2
        ax.set_xlim(mid[0] - span, mid[0] + span)
        ax.set_ylim(mid[1] - span, mid[1] + span)
        ax.set_zlim(mid[2] - span, mid[2] + span)

        ax.set_axis_off()
        ax.view_init(elev=25, azim=45)
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor=fig.get_facecolor())
        plt.close()
        print(f"Screenshot saved to {filename}")

    @staticmethod
    def preview(vine: CosmicVine):
        """
        Opens an interactive 3D viewer via trimesh's built-in renderer.

        Combines all blocks into a single mesh before passing to the viewer.
        Requires a display (not available in headless environments).
        """
        mesh = vine.get_combined_mesh()
        mesh.show()


# --- Entry Point ---
if __name__ == "__main__":
    # Initialize the engine
    vine = CosmicVine(block_size=1.0)

    # Execute growth logic
    vine.grow(iterations=100, max_rotation_rad=np.pi / 4)

    # Export
    VineExporter.save(vine, 'formx_cosmic_vine.obj')
    VineExporter.screenshot(vine, 'formx_cosmic_vine.png')
    VineExporter.preview(vine)
