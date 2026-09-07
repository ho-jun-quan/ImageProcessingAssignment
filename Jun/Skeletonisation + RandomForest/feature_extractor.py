"""
feature_extractor.py — Turn skeletonised particle tracks into inputs for the
Random Forest classifier.

The primary representation is the skeleton IMAGE itself: each track skeleton is
resized to a fixed grid (config.SKELETON_IMG_SIZE) and flattened into a binary
pixel vector via ``skeleton_to_vector``. The legacy 12-element geometric
``extract_features`` vector is retained for reference/analysis.
"""

import cv2
import numpy as np
from scipy import ndimage

import config


def skeleton_to_vector(skeleton, size=None):
    """
    Convert a track skeleton into a flattened, fixed-size binary pixel vector.

    This is the representation the Random Forest is trained on: the resized
    skeleton image pixels themselves.

    Parameters
    ----------
    skeleton : np.ndarray
        Skeleton array (bool or 0/255). Any 2-D size is accepted.
    size : tuple (height, width) or None
        Target grid. Defaults to ``config.SKELETON_IMG_SIZE``.

    Returns
    -------
    vector : np.ndarray, shape (height * width,)
        Flattened binary skeleton image (values in {0.0, 1.0}).
    """
    if size is None:
        size = config.SKELETON_IMG_SIZE
    h, w = size

    skel = np.asarray(skeleton)
    if skel.ndim > 2:
        skel = skel[..., 0]
    skel = (skel > 0).astype(np.uint8) * 255

    # INTER_AREA preserves thin structures better when downsizing
    resized = cv2.resize(skel, (w, h), interpolation=cv2.INTER_AREA)
    binary = (resized > 0).astype(np.float32)
    return binary.flatten()


def _count_neighbours(skeleton):
    """
    For each True pixel in the skeleton, count how many of its 8-connected
    neighbours are also True.

    Returns an array of the same shape with neighbour counts at skeleton
    pixels (0 elsewhere).
    """
    # Convert to uint8 for convolution
    skel = skeleton.astype(np.uint8)

    # 3×3 kernel counting all 8 neighbours (center excluded)
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    neighbour_count = ndimage.convolve(skel, kernel, mode="constant", cval=0)

    # Only keep counts at skeleton pixels
    return neighbour_count * skel


def _get_endpoints(skeleton):
    """
    Find endpoint pixels (exactly 1 neighbour) in the skeleton.

    Returns array of (row, col) coordinates.
    """
    nc = _count_neighbours(skeleton)
    endpoints = np.argwhere((nc == 1) & skeleton)
    return endpoints


def _get_branch_points(skeleton):
    """
    Find branch/junction pixels (≥3 neighbours) in the skeleton.

    Returns array of (row, col) coordinates.
    """
    nc = _count_neighbours(skeleton)
    branch_pts = np.argwhere((nc >= 3) & skeleton)
    return branch_pts


def _trace_skeleton_path(skeleton):
    """
    Trace the skeleton from one endpoint to another, returning an ordered
    list of (row, col) coordinates along the path.

    If the skeleton has branches, traces the longest path between any two
    endpoints.
    """
    endpoints = _get_endpoints(skeleton)
    skel_pixels = np.argwhere(skeleton)

    if len(skel_pixels) == 0:
        return np.array([])

    if len(endpoints) < 2:
        # No clear endpoints; just return all skeleton pixels in order
        return skel_pixels

    # Use BFS from the first endpoint to find the farthest endpoint,
    # then trace the path
    start = tuple(endpoints[0])
    skel_set = set(map(tuple, skel_pixels))

    # BFS to find distances from start
    from collections import deque
    visited = {start: None}
    queue = deque([start])

    while queue:
        current = queue.popleft()
        r, c = current
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in skel_set and nb not in visited:
                    visited[nb] = current
                    queue.append(nb)

    # Find the farthest endpoint from start
    max_dist_endpoint = start
    max_dist = 0
    for ep in endpoints:
        ep_tuple = tuple(ep)
        if ep_tuple in visited:
            # Trace back to measure distance
            dist = 0
            node = ep_tuple
            while visited[node] is not None:
                node = visited[node]
                dist += 1
            if dist > max_dist:
                max_dist = dist
                max_dist_endpoint = ep_tuple

    # Now do BFS from max_dist_endpoint to find the true longest path
    end = max_dist_endpoint
    visited2 = {end: None}
    queue2 = deque([end])

    while queue2:
        current = queue2.popleft()
        r, c = current
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                nb = (r + dr, c + dc)
                if nb in skel_set and nb not in visited2:
                    visited2[nb] = current
                    queue2.append(nb)

    # Find farthest point from end
    farthest = end
    max_d = 0
    for ep in endpoints:
        ep_tuple = tuple(ep)
        if ep_tuple in visited2:
            dist = 0
            node = ep_tuple
            while visited2[node] is not None:
                node = visited2[node]
                dist += 1
            if dist > max_d:
                max_d = dist
                farthest = ep_tuple

    # Trace path from farthest back to end
    path = []
    node = farthest
    while node is not None:
        path.append(node)
        node = visited2.get(node)

    return np.array(path)


def _compute_curvature(path, step=5):
    """
    Compute local curvature along a traced skeleton path.

    Curvature at each point is measured as the angle change between
    consecutive segments of length `step`.

    Returns (total_curvature, max_curvature).
    """
    if len(path) < 2 * step + 1:
        return 0.0, 0.0

    curvatures = []
    for i in range(step, len(path) - step):
        # Vector from (i-step) to i
        v1 = path[i] - path[i - step]
        # Vector from i to (i+step)
        v2 = path[i + step] - path[i]

        v1 = v1.astype(float)
        v2 = v2.astype(float)

        # Compute angle between the two vectors
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)

        if norm1 < 1e-6 or norm2 < 1e-6:
            continue

        cos_angle = np.clip(np.dot(v1, v2) / (norm1 * norm2), -1.0, 1.0)
        angle = np.arccos(cos_angle)
        curvatures.append(angle)

    if len(curvatures) == 0:
        return 0.0, 0.0

    total_curvature = float(np.sum(curvatures))
    max_curvature = float(np.max(curvatures))

    return total_curvature, max_curvature


def extract_features(skeleton, binary_mask, bbox):
    """
    Extract a 12-element feature vector from a single track's skeleton
    and binary mask.

    Parameters
    ----------
    skeleton : np.ndarray (bool)
        1-pixel-wide skeleton of the track.
    binary_mask : np.ndarray (uint8)
        Binary mask of the track (0 or 255).
    bbox : tuple (x, y, w, h)
        Bounding box of the track in the original frame.

    Returns
    -------
    features : np.ndarray, shape (12,)
        Feature vector.
    """
    x, y, w, h = bbox

    # --- Feature 1: Skeleton length (number of True pixels) ---
    skeleton_length = float(np.sum(skeleton))

    # --- Feature 2: Bounding box aspect ratio ---
    bbox_aspect_ratio = float(w) / max(float(h), 1.0)

    # --- Feature 3: Track thickness (contour area / skeleton length) ---
    contour_area = float(np.sum(binary_mask > 0))
    track_thickness = contour_area / max(skeleton_length, 1.0)

    # --- Feature 4: Straightness (end-to-end distance / path length) ---
    endpoints = _get_endpoints(skeleton)
    if len(endpoints) >= 2:
        # Euclidean distance between the two most distant endpoints
        dists = []
        for i in range(len(endpoints)):
            for j in range(i + 1, len(endpoints)):
                d = np.linalg.norm(endpoints[i] - endpoints[j])
                dists.append(d)
        end_to_end = max(dists) if dists else 0.0
    elif len(endpoints) == 1:
        # Single endpoint — find farthest skeleton pixel
        skel_pixels = np.argwhere(skeleton)
        if len(skel_pixels) > 0:
            dists = np.linalg.norm(skel_pixels - endpoints[0], axis=1)
            end_to_end = float(np.max(dists))
        else:
            end_to_end = 0.0
    else:
        end_to_end = 0.0

    straightness = end_to_end / max(skeleton_length, 1.0)
    straightness = min(straightness, 1.0)  # Cap at 1.0

    # --- Features 5 & 6: Total and max curvature ---
    path = _trace_skeleton_path(skeleton)
    total_curvature, max_curvature = _compute_curvature(path)

    # --- Feature 7: Number of branch points ---
    branch_points = _get_branch_points(skeleton)
    num_branch_points = float(len(branch_points))

    # --- Feature 8: Number of endpoints ---
    num_endpoints = float(len(endpoints))

    # --- Feature 9: Branch ratio ---
    # Ratio of total skeleton pixels NOT on the main path to main path length
    main_path_length = float(len(path)) if len(path) > 0 else skeleton_length
    branch_pixels = skeleton_length - main_path_length
    branch_ratio = max(branch_pixels, 0.0) / max(main_path_length, 1.0)

    # --- Feature 10: Contour area ---
    # (already computed above)

    # --- Feature 11: Skeleton-to-area ratio ---
    skeleton_area_ratio = skeleton_length / max(contour_area, 1.0)

    # --- Feature 12: Bounding box fill ratio ---
    bbox_area = float(w * h)
    bbox_fill_ratio = contour_area / max(bbox_area, 1.0)

    features = np.array([
        skeleton_length,
        bbox_aspect_ratio,
        track_thickness,
        straightness,
        total_curvature,
        max_curvature,
        num_branch_points,
        num_endpoints,
        branch_ratio,
        contour_area,
        skeleton_area_ratio,
        bbox_fill_ratio,
    ], dtype=np.float64)

    return features
