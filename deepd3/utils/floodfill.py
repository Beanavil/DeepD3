import gc
import tifffile
import numpy as np
import skimage as sk
import scipy.ndimage as ndi
from collections import deque
from scipy.spatial.distance import cdist


def get_neighborhood(z, y, x, shape):
    """Gets the neightboring voxels of an (x,y,z) point in a 3D image of shape 'shape'.

    Args:
        x (int): coordinate of the point to get the neighborhood for in the OX axis.
        y (int): coordinate of the point to get the neighborhood for in the OY axis.
        z (int): coordinate of the point to get the neighborhood for in the OZ axis.
        shape (array): 1D array with three elements, storing the Z, Y and X dimensions of the image
                       (strictly in that order).

    Returns:
        array: 1D array with all 27 neighbors of the given 3D point.
    """
    z, y, x = int(round(z)), int(round(y)), int(round(x))
    neighbors = []
    for i in range(-1, 2):
        for j in range(-1, 2):
            for k in range(-1, 2):
                if i + j + k == 0:
                    continue
                nz, ny, nx = z + k, y + j, x + i
                if 0 <= nz < shape[0] and 0 <= ny < shape[1] and 0 <= nx < shape[2]:
                    neighbors.append((nz, ny, nx))
    return neighbors


def get_medoid(mask, max_coords=10000):
    """Gets the medoid of a mask.

    Args:
        mask (array): 3D array storing the binary mask to calculate the medoid for.
        max_cords (int): Max number of mask elements to consider for medoid, ensuring controlled memory use. Default to 10000.
    Returns:
        tuple: 3D coordinates of the medoid.
    """
    coords = np.argwhere(mask > 0)
    if len(coords) <= max_coords:
        dists = cdist(coords, coords).astype(np.float32)
        medoid_index = np.argmin(dists.sum(axis=1))
        return coords[medoid_index]
    random_idxs = np.random.choice(len(coords), max_coords, replace=False)
    random_coords = coords[random_idxs]
    dists = cdist(random_coords, coords).astype(np.float32)
    medoid_index = np.argmin(dists.sum(axis=1))
    return random_coords[medoid_index]


def floodfill_impl(
    img, initial_mask, dimg, min_threshold, forbidden_mask=None, dendrite_ff=False
):
    """Flood-fills spine mask labeled with 'label_id' across Z slices from sparse mask annotations,
    using intensity and 3D connectivity. Optionally excludes 'forbidden' voxels.

    Args:
        img (array): 3D image stack (Z, Y, X).
        initial_mask (array): 3D binary mask for current spine/dendrite.
        forbidden_mask (array): 3D binary mask where fill is not allowed (e.g. dendrite).
        alpha (float): parameter that determines the cut on intensity for a voxel to be chosen as seed.
                       Default to 0.85. Should belong to [0, 1].

    Returns:
        array: 3D binary mask after flood-fill.
    """
    # Sanity check
    if forbidden_mask is None:
        forbidden_mask = np.zeros_like(img, dtype=bool)

    # Binary masks to control the visited neighbors and the mask bits added
    visited_mask = np.zeros_like(img, dtype=bool)
    final_mask = np.zeros_like(img, dtype=bool)
    final_mask[initial_mask] = True

    # Spines sometimes have complex shapes, so the centroid may fall in background. Use a center
    # calculated from the weighted medians instead.
    # When floodfilling the dendrite, use max brightness pixel.
    zyx_coords = np.argwhere(initial_mask)
    initial_intensities = img[initial_mask]
    cz, cy, cx = (
        zyx_coords[np.argmax(initial_intensities)]
        if dendrite_ff
        else get_medoid(initial_mask)
    )

    # Intensities vary along the spine, and we at least want to select the current masked pixels.
    # For finding a good threshold, we first remove the outliers (e.g. if some background pixels where mistakenly
    # masked, those will have ~0 intensity, much lower than spine pixels), and then we set as threshold the median
    # intensity from all the intensities left.
    p_low, p_high = np.percentile(initial_intensities, [5, 95])
    cleaned_intensities = initial_intensities[
        (initial_intensities > p_low) & (initial_intensities < p_high)
    ]
    if cleaned_intensities.size == 0:
        threshold = np.median(initial_intensities)
    else:
        threshold = np.median(cleaned_intensities)

    # Additionally, for a more robust floodfill search, we include the pixel with an intensity closest to the median
    # as seed. This avoids that if the centroid of the mask falls in a background pixel (e.g. if the mask included pixels
    # that are actually background) then we have at least a seed that is in the actual spine.
    closest_voxel = np.argmin(np.abs(initial_intensities - threshold))
    mz, my, mx = zyx_coords[closest_voxel]

    # Never allow a too low threshold brightness for corner cases in which spines are not very bright.
    if not dendrite_ff:
        threshold = max(threshold, min_threshold, img[cz, cy, cx], img[mz, my, mx])
    else:
        threshold = max(threshold, min_threshold)

    # Sanity check
    if not img[mz, my, mx] or not threshold or not img[cz, cy, cx]:
        return final_mask

    # Initialize queue with centroid and pixel with median intensity
    queue = deque()
    queue.extend(get_neighborhood(cz, cy, cx, img.shape))
    queue.extend(get_neighborhood(mz, my, mx, img.shape))

    # First derivative values for centroid's brightness interpolation.
    img_z_grad = dimg / 2

    # Loop over neighbors.
    while queue:
        current = queue.popleft()
        z, y, x = current

        # Skip if voxel is already visited or forbidden.
        if visited_mask[z, y, x]:
            continue
        else:
            visited_mask[z, y, x] = True
        if forbidden_mask[:, y, x].any():
            continue

        # If voxel is not bright enough, we might have encountered a boundary.
        # Interpolate threshold from median brightness's by decreasing the threshold when we move
        # to less bright areas.
        delta_z = abs(mz - z)
        z_grad = img_z_grad[z, y, x]
        taylor_1_threshold = threshold + z_grad * delta_z
        if img[z, y, x] < taylor_1_threshold:
            continue

        # Add accepted voxel to the mask and its neighbors to the queue.
        final_mask[z, y, x] = True
        neighbors = get_neighborhood(z, y, x, img.shape)
        for nz, ny, nx in neighbors:
            queue.append((nz, ny, nx))

    return final_mask


def floodfill(stack, dendrite_mask, spines_mask):
    """Flood-fills the spines' masks from a sparsely annotated image across Z slices. Uses the
       dendrite mask as 'black list' for avoiding flood-filling past the dendrite boundary.

    Args:
        stack (array): original 3D image stack (Z, Y, X) in array representation
        spines_mask (array): 3D binary mask for all the spines
        dendrite_mask (array): 3D binary mask for the dendrite

    Returns:
        array: 3D binary mask of spines after flood-filling
    """
    # Sanity check
    if not dendrite_mask.any() or not spines_mask.any():
        return spines_mask

    # Get all spine labels and excluding the background (label = 0).
    L, _ = ndi.label(spines_mask)
    labels = np.unique(L)
    labels = labels[labels != 0]

    # Initialize final mask to accumulate filled results.
    final_spines_mask = np.zeros_like(L, dtype=bool)

    # Get image derivative (intensity change rates) in OZ.
    denoised = np.memmap(
        "denoised.dat", dtype=stack.dtype, mode="w+", shape=stack.shape
    )
    ndi.median_filter(stack, size=3, output=denoised)
    threshold = sk.filters.threshold_triangle(denoised)
    dimg = np.zeros_like(denoised)
    np.multiply(denoised, denoised > threshold, out=dimg)
    dimg = ndi.prewitt(dimg, axis=0)

    # Get dendrite mask and extend it over the whole Z axis for avoiding floodfilling spines into dendrite.
    dendrite_mask_broad = np.any((dendrite_mask > 0), axis=0)
    dendrite_mask_broad = np.broadcast_to(dendrite_mask_broad, dendrite_mask.shape)
    spines_mask_broad = np.any(spines_mask, axis=0)
    spines_mask_broad = np.broadcast_to(spines_mask_broad, spines_mask.shape)
    dendrite_mask_ff = floodfill_impl(
        img=stack,
        initial_mask=(dendrite_mask > 0),
        forbidden_mask=spines_mask_broad,
        dimg=dimg,
        min_threshold=np.percentile(stack[dendrite_mask > 0], 25),
        dendrite_ff=True,
    )

    forbidden_mask = (dendrite_mask_broad | dendrite_mask_ff) & ~spines_mask

    # Delete unused arrays.
    del denoised, dendrite_mask_broad, spines_mask_broad, dendrite_mask_ff
    gc.collect()

    # Global-pov threshold.
    min_threshold = np.percentile(stack[spines_mask > 0], 50)

    # Loop through all spines.
    for label_id in labels:
        mask = L == label_id
        # Floodfill and acumulate into final mask.
        final_spines_mask |= floodfill_impl(
            img=stack,
            initial_mask=mask,
            forbidden_mask=forbidden_mask,
            dimg=dimg,
            min_threshold=min_threshold,
        )

    return final_spines_mask


def floodfill_stacks(fn):
    """Floodfills the spines' masks from all the stacks and writes the images back to disk

    Args:
        fn (str): A list of paths to the data stacks. They should be raw images, d3set is not supported yet.
    """
    for i in range(len(fn)):
        img = tifffile.memmap(fn[i]["img"])
        d_mask = tifffile.memmap(fn[i]["d_mask"])
        s_masks = tifffile.memmap(fn[i]["s_masks"])
        spines_mask_data_ff = floodfill(
            stack=img,
            dendrite_mask=d_mask,
            spines_mask=s_masks,
        )
        tifffile.imwrite(fn[i]["s_masks"], spines_mask_data_ff.astype(np.uint8) * 255)
        print(f"Spines masks from stack {i} ({fn[i]['img']}) have been floodfilled")
