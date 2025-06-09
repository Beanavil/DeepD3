import numpy as np
from collections import deque
from skimage.measure import label
import tifffile


def get_neighborhood(x, y, z, shape):
    """Gets the neightboring voxels of an (x,y,z) point in a 3D image of shape 'shape'.

    Args:
        x (int): coordinate of the point to get the neightborhood for in the OX axis.
        y (int): coordinate of the point to get the neightborhood for in the OY axis.
        z (int): coordinate of the point to get the neightborhood for in the OZ axis.
        shape (array): 1D array with three elements, storing the Z, Y and X dimensions of the image
                       (strictly in that order).

    Returns:
        array: 1D array with all 27 neighbors of the given 3D point.
    """
    x, y, z = int(round(x)), int(round(y)), int(round(z))
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i + j + k == 0:
                    continue
                nx, ny, nz = x + i, y + j, z + k
                if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                    neighbors.append((nx, ny, nz))
    return neighbors


def floodfill_impl(img, initial_mask, max_voxels=None, forbidden_mask=None, alpha=0.8):
    """Flood-fills a spine region across Z slices from sparse mask annotations,
    using intensity and 3D connectivity. Optionally excludes 'forbidden' voxels.

    Args:
        img (array): 3D image stack (Z, Y, X).
        initial_mask (array): 3D binary mask for current spine.
        max_voxels (int): optional cap on max number of voxels to include.
        forbidden_mask (array): 3D binary mask where fill is not allowed (e.g. dendrite).
        alpha (float): parameter that determines the cut on intensity for a voxel to be chosen as seed.
                       Default ot 0.8.

    Returns:
        array: 3D binary mask after flood-fill.
    """
    final_mask = np.zeros_like(img, dtype=bool)
    final_mask[initial_mask] = True

    # Use a high-intensity voxel from the mask as seed
    zyx_coords = np.argwhere(initial_mask)
    seed_z, seed_y, seed_x = zyx_coords[np.argmax(img[initial_mask])]
    seed_intensity = float(img[seed_z, seed_y, seed_x])
    threshold = min(seed_intensity * alpha,
                    np.percentile(img[initial_mask], 85))

    # Initialize queue
    queue = deque()
    queue.extend(get_neighborhood(seed_x, seed_y, seed_z, img.shape))

    # Loop over neightbors
    while queue:
        x, y, z = queue.popleft()
        # Bounds check
        if not (
            0 <= z < img.shape[0] and 0 <= y < img.shape[1] and 0 <= x < img.shape[2]
        ):
            continue
        # Already visited or forbidden voxels are discarded
        if final_mask[z, y, x]:
            continue
        if forbidden_mask is not None and forbidden_mask[z, y, x]:
            continue
        # Intensity check. If intense enough, the neightbor is added to the mask
        if img[z, y, x] > threshold:
            final_mask[z, y, x] = True
            neighbors = get_neighborhood(x, y, z, img.shape)
            for nx, ny, nz in neighbors:
                if forbidden_mask is not None and forbidden_mask[nz, ny, nx]:
                    continue
                if final_mask[nz, ny, nx]:
                    continue
                queue.append((nx, ny, nz))
            if max_voxels is not None and np.count_nonzero(final_mask) > max_voxels:
                print(
                    f"Floodfill aborted: number of voxels added ({np.count_nonzero(final_mask)}) exceeded max_voxels ({max_voxels})"
                )
                break
    return final_mask


def floodfill(stack, dendrite_mask, spines_masks):
    """Flood-fills the spines' masks from a sparsely annotated image across Z slices. Uses the
       dendrite mask as 'black list' for avoiding flood-filling past the dendrite boundary.

    Args:
        stack (array): original 3D image stack (Z, Y, X) in array representation
        spines_masks (array): 3D binary mask for all the spines
        dendrite_mask (array): 3D binary mask for the dendrite

    Returns:
        array: 3D binary mask of spines after flood-filling
    """
    # Get all spine labels and excluding the background (label = 0)
    L = label(spines_masks)
    labels = np.unique(L)
    labels = labels[labels != 0]

    # Initialize final mask to accumulate filled results
    final_mask = np.zeros_like(L, dtype=bool)

    # Store diff to analyze floodfill result later
    diff_map = np.zeros_like(L, dtype=np.int8)

    # Extend dendrite mask over the whole Z axis
    dendrite_mask_combined = np.any(dendrite_mask, axis=0)
    dendrite_mask_broad = np.broadcast_to(
        dendrite_mask_combined, dendrite_mask.shape)

    # Loop through all spines
    for label_id in labels:
        mask = L == label_id
        filled = floodfill_impl(
            img=stack, initial_mask=mask, forbidden_mask=dendrite_mask_broad
        )
        # Accumulate into final mask
        final_mask |= filled
        # Compute accumulated difference map
        diff = filled.astype(int) - mask.astype(int)
        diff_map += diff.astype(np.int8)

    return final_mask, diff_map


def floodfill_stacks(fn):
    """Floodfills the spines' masks from all the stacks and writes the images back to disk

    Args:
        fn (str): A list of paths to the data stacks. They should be raw images, d3set is not supported yet.
    """
    for i in range(len(fn)):
        img = tifffile.imread(fn[i]['img'])
        d_mask = tifffile.imread(fn[i]['d_mask'])
        s_masks = tifffile.imread(fn[i]['s_masks'])
        spines_mask_data_ff, diff_map = floodfill(
            stack=img,
            dendrite_mask=d_mask,
            spines_masks=s_masks,
        )
        # Only write back to disk if floodfilling made any difference
        if np.any(diff_map):
            tifffile.imwrite(fn[i]['s_masks'], spines_mask_data_ff)
            print(
                f"Spines masks from stack {i} were floodfilled with {np.sum(diff_map)} voxels")
