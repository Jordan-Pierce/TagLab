import os
from time import time

import numpy as np

import Metashape

# Get the Metashape License stored in the environmental variable
Metashape.License().activate(os.getenv("METASHAPE_LICENSE"))

# Check that the Metashape version is compatible with this script
compatible_major_version = "2.0"
found_major_version = ".".join(Metashape.app.version.split('.')[:2])
if found_major_version != compatible_major_version:
    raise Exception("Incompatible Metashape version: {} != {}".format(found_major_version, compatible_major_version))


# ----------------------------------------------------------------------------------------------------------------------
# Functions
# ----------------------------------------------------------------------------------------------------------------------
def get_depths(chunk):
    """

    """
    print("NOTE: Calculating depths")

    depth_maps = []
    depths = []

    for camera in chunk.cameras:
        try:
            depth_map = chunk.depth_maps[camera]
            depth = depth_map.image()

            depth_maps.append(depth_map)
            depths.append(depth)
        except:
            continue

    cameras = [c.label for c in chunk.cameras]
    depth_maps = dict(zip(cameras, depth_maps))
    depths = dict(zip(cameras, depths))

    return depth_maps, depths


def distance(point1, point2):
    """

    """
    # Convert points to NumPy arrays for easy computation
    point1 = np.array(point1)
    point2 = np.array(point2)

    # Calculate the Euclidean distance
    dist = np.sqrt(np.sum((point2 - point1)**2))

    return dist


def find_cameras(pixel, chunk, depth_maps, depths):
    """

    """

    positions = []

    T = chunk.transform.matrix
    orthomosaic = chunk.orthomosaic

    # Pixel coordinates of point p on orthomosaic
    x, y = pixel

    # Point p 2D coordinates in ortho CS
    X, Y = (orthomosaic.left + orthomosaic.resolution * x, orthomosaic.top - orthomosaic.resolution * y)

    if chunk.elevation:

        # Access the dem
        dem = chunk.elevation

        # Altitude in dem.crs (supposing dem.crs  = ortho.crs)
        Z = dem.altitude(Metashape.Vector((X, Y)))

        # X, Y, Z  point p 3D coordinates  in ortho CS
        if orthomosaic.crs.name[0:17] == 'Local Coordinates':
            # Point p in internal coordinate system for case of Local CS
            p = T.inv().mulp(orthomosaic.projection.matrix.inv().mulp(Metashape.Vector((X, Y, Z))))
        else:
            # Point p in internal coordinate system
            p = T.inv().mulp(orthomosaic.crs.unproject(Metashape.Vector((X, Y, Z))))

    else:
        print("ERROR: No DEM found")
        return

    for camera in chunk.cameras:

        # If the point doesn't project, skip
        if not camera.project(p):
            continue

        u = camera.project(p).x  # u pixel coordinates in camera
        v = camera.project(p).y  # v pixel coordinates in camera

        # Failed the first test, in that the point in not in the image at all
        if u < 0 or u > camera.sensor.width or v < 0 or v > camera.sensor.height:
            continue

        try:
            # -----------------
            # Obstruction test
            # -----------------

            # Create a Metashape.Vector with pixel coordinates (u, v)
            Q = Metashape.Vector([u, v])

            # Retrieve the depth map for the current camera
            depth_map = depth_maps[camera.label]

            # Project the 2D pixel coordinates (u, v) back to 3D world coordinates (x, y)
            x, y = depth_map.calibration.project(camera.calibration.unproject(Q))

            # Retrieve the depth value at the calculated (x, y) position in the depth map
            depth = depths[camera.label]
            d = depth[int(x), int(y)][0]

            # Calculate the 3D point on the ray from the camera center through the pixel coordinates (u, v)
            ray = camera.unproject(Q) - camera.center
            ray /= ray.norm()
            q = camera.center + (d / camera.transform.inv().mulv(ray).z) * ray

            # Check if the distance between the original 3D point (p) and the calculated 3D point (q)
            # exceeds a threshold; raise an exception.
            delta = np.abs(distance(camera.center - p, camera.center - q))

            if delta > 1:
                raise Exception

        # If any exception occurs in the try block, skip to the next iteration of the loop
        except:
            continue

        dist = distance(camera.center, p)
        positions.append([camera.label, u, v, dist])

    return np.array(positions)


# Create a metashape doc object
doc = Metashape.Document()
doc.open(r"B:\CoralNet-Toolbox\Data\4352\sfm\2023-09-18_09-59-18\project.psx")

# Get the chunk
chunk = doc.chunk

# Get depth maps and depth before, calculate only once
depth_maps, depth = get_depths(chunk)

# Pixel on orthomosaic
pixel = 1020, 2310

positions = find_cameras(pixel, chunk, depth_maps, depth)
top_cameras = positions[np.argsort(positions[:, 3])[:5]]

print(f"NOTE: Found {len(positions)} with pixel {pixel} in view")
print(f"NOTE: Top 5 cameras are")
for top_camera in top_cameras:
    print(top_camera[0], top_camera[-1])

print("Done.")
