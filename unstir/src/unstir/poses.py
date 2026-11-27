"""Relative camera poses of the lookup photo and its matched images.

Every matched pair gives an essential matrix, so a rotation and the direction the
camera moved, but not how far. All pairs share the keypoints of the query: a query
keypoint triangulated in two pairs gives two depths along the same ray, and their
ratio puts both pairs at one scale. The result is one frame around the query, up
to one global scale.

No camera has known intrinsics. The focal length of the query comes from its EXIF
data when it has one, the others are a guess with the principal point at the
centre. Architectural plates were often taken with a shifted lens, so the poses
are a sketch, not a survey.
"""

from dataclasses import dataclass

import cv2
import numpy as np

FOCAL = 1.0
"""Focal length of a camera without EXIF data, times the long side of its image
(about a 36 mm lens on 35 mm film)."""
MIN_POINTS = 15
"""Triangulated points a pair needs to give a pose."""
MIN_PARALLAX = 1.5
"""Median angle in degrees between the two rays to a point. Below it the cameras
stood at about the same place, and the direction of the move is noise."""
MIN_SHARED = 5
"""Query keypoints a pair must share with the placed pairs to get their scale."""
MAX_POINTS = 2000
FAR = 5.0
"""Points farther from the query than this times the median distance are left out."""

FAR_POINT = 1e4
"""`recoverPose` drops points farther than this times the move. Its default of 50
drops everything when two cameras stood at almost the same place, and then such
a camera would not even get its rotation."""

UP = np.array([0.0, -1.0, 0.0])
"""Up in a camera frame (OpenCV: x right, y down, z forward)."""
FORWARD = np.array([0.0, 0.0, 1.0])


@dataclass
class View:
    """One matched image, as the web page names it."""

    key: str
    row: int
    name: str
    collection: str


@dataclass
class Pair:
    """The pose of one matched image relative to the query, for a move of length 1."""

    view: View
    rotation: np.ndarray
    """Query camera frame to this camera's frame: `x = R x_query + t`."""
    move: np.ndarray
    keypoints: np.ndarray
    """The query keypoint of each point in `points`."""
    points: np.ndarray
    """Triangulated points, `(n, 3)`, in the query camera frame."""
    parallax: float
    shape: tuple[int, int]
    focal: float

    @property
    def centre(self) -> np.ndarray:
        return -self.rotation.T @ self.move

    def depths(self) -> zip:
        """(query keypoint, depth in the query camera) of each point."""
        return zip(self.keypoints.tolist(), self.points[:, 2].tolist(), strict=True)


def focal_35mm(focal: float | None, shape: tuple[int, ...]) -> float:
    """Focal length in pixels from a 35 mm equivalent one (36 mm wide film), or
    the guess when there is none."""
    side = max(shape[:2])
    return focal / 36 * side if focal else FOCAL * side


def normalise(points: np.ndarray, shape: tuple[int, ...], focal: float) -> np.ndarray:
    height, width = shape[:2]
    return ((points - [width / 2, height / 2]) / focal).astype(np.float64)


def relative(
    view: View,
    pts_q: np.ndarray,
    pts_c: np.ndarray,
    idx_q: np.ndarray,
    shape_q: tuple[int, ...],
    shape_c: tuple[int, ...],
    focal_q: float,
    threshold_px: float,
) -> Pair | None:
    """The pose of the matched image, or None when the matches do not give one."""
    if len(pts_q) < MIN_POINTS:
        return None
    focal_c = focal_35mm(None, shape_c)
    nq, nc = normalise(pts_q, shape_q, focal_q), normalise(pts_c, shape_c, focal_c)
    threshold = 2 * threshold_px / (focal_q + focal_c)
    essential, mask = cv2.findEssentialMat(
        nq, nc, np.eye(3), cv2.USAC_MAGSAC, 0.999, threshold, 10_000
    )
    if essential is None or mask is None:
        return None
    good, rotation, move, mask, _ = cv2.recoverPose(
        essential[:3], nq, nc, np.eye(3), distanceThresh=FAR_POINT, mask=mask
    )
    if good < MIN_POINTS:
        return None
    keep = mask.ravel() > 0
    move = move.ravel()
    found = cv2.triangulatePoints(
        np.eye(3, 4), np.hstack([rotation, move[:, None]]), nq[keep].T, nc[keep].T
    )
    points = (found[:3] / found[3]).T
    front = (points[:, 2] > 0) & ((points @ rotation.T + move)[:, 2] > 0)
    points = points[front]
    if len(points) < MIN_POINTS:
        return None
    centre = -rotation.T @ move
    rays_q = points / np.linalg.norm(points, axis=1, keepdims=True)
    rays_c = points - centre
    rays_c /= np.linalg.norm(rays_c, axis=1, keepdims=True)
    angles = np.degrees(np.arccos(np.clip((rays_q * rays_c).sum(axis=1), -1, 1)))
    return Pair(
        view=view,
        rotation=rotation,
        move=move,
        keypoints=idx_q[keep][front],
        points=points,
        parallax=float(np.median(angles)),
        shape=(int(shape_c[0]), int(shape_c[1])),
        focal=focal_c,
    )


def scales(pairs: list[Pair]) -> dict[int, float]:
    """The scale of each pair (by index in `pairs`) that could be joined to the
    others through shared query keypoints. The pair with the most points sets it."""
    usable = sorted(
        (i for i, p in enumerate(pairs) if p.parallax >= MIN_PARALLAX),
        key=lambda i: -len(pairs[i].points),
    )
    if not usable:
        return {}
    found = {usable[0]: 1.0}
    depths = dict(pairs[usable[0]].depths())
    remaining = usable[1:]

    def shared(i: int) -> list[float]:
        return [depths[k] / z for k, z in pairs[i].depths() if k in depths]

    while remaining:
        best = max(remaining, key=lambda i: len(shared(i)))
        ratios = shared(best)
        if len(ratios) < MIN_SHARED:
            break
        scale = found[best] = float(np.median(ratios))
        depths.update(
            (k, z * scale) for k, z in pairs[best].depths() if k not in depths
        )
        remaining.remove(best)
    return found


def fov(shape: tuple[int, ...], focal: float) -> float:
    """Vertical field of view in degrees."""
    return float(np.degrees(2 * np.arctan(shape[0] / 2 / focal)))


def scene(pairs: list[Pair], shape_q: tuple[int, ...], focal_q: float) -> dict:
    """The cameras and points as the web page draws them: y up, the query at the
    origin and looking along -z (as a three.js camera does).

    A camera's `status` is `placed` (joined to the others at one scale),
    `unscaled` (right direction, distance unknown) or `same-spot` (too little
    parallax: it stood about where the query stood, only its direction counts).
    """
    found = scales(pairs)
    ups = [UP, *(p.rotation.T @ UP for p in pairs)]
    up = np.mean(ups, axis=0)
    up /= np.linalg.norm(up)
    forward = FORWARD - (FORWARD @ up) * up
    forward /= np.linalg.norm(forward)
    basis = np.stack([np.cross(forward, up), up, -forward])
    typical = float(np.median(list(found.values()))) if found else 1.0

    def status(i: int) -> str:
        if i in found:
            return "placed"
        return "same-spot" if pairs[i].parallax < MIN_PARALLAX else "unscaled"

    def camera(view: View | None, position, rotation, shape, focal, state) -> dict:
        return {
            "key": view.key if view else None,
            "row": view.row if view else None,
            "name": view.name if view else "your photo",
            "collection": view.collection if view else None,
            "status": state,
            "position": rounded(basis @ position),
            "forward": rounded(basis @ rotation.T @ FORWARD),
            "up": rounded(basis @ rotation.T @ UP),
            "fov": round(fov(shape, focal), 1),
            "aspect": round(shape[1] / shape[0], 3),
        }

    cameras = [camera(None, np.zeros(3), np.eye(3), shape_q, focal_q, "query")]
    for i, pair in enumerate(pairs):
        state = status(i)
        distance = {"placed": found.get(i), "unscaled": typical, "same-spot": 0.0}
        cameras.append(
            camera(
                pair.view,
                pair.centre * distance[state],
                pair.rotation,
                pair.shape,
                pair.focal,
                state,
            )
        )

    points = (
        np.concatenate([pairs[i].points * s for i, s in found.items()])
        if found
        else np.zeros((0, 3))
    )
    if len(points):
        distances = np.linalg.norm(points, axis=1)
        points = points[distances <= FAR * np.median(distances)]
        points = points[:: max(1, len(points) // MAX_POINTS)]
    return {
        "cameras": cameras,
        "points": [rounded(p) for p in points @ basis.T],
    }


def rounded(vector: np.ndarray) -> list[float]:
    return [round(float(v), 4) for v in vector]
