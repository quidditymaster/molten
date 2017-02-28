
import numpy as np
import pandas as pd
import scipy.ndimage.morphology
import skimage.measure
import stl

def cloud2mesh(
        cloud,
        nside,
        bounds=(-1, 1, -1, 1, -1, 1),
        edge_behavior = "clip",
        count_threshold=1,
        closure=3,
):
    grid = np.zeros((nside, nside, nside))
    xmin, xmax, ymin, ymax, zmin, zmax = bounds
    
    x = cloud["x"].values
    y = cloud["y"].values
    z = cloud["z"].values
    
    if edge_behavior == "clip":
        xi = np.clip(x, xmin, xmax)
        yi = np.clip(y, ymin, ymax)
        zi = np.clip(z, zmin, zmax)
    elif edge_behavior == "mask":
        mask = (x <= xmax)&(x >= xmin)
        mask &= (y <= ymax)&(y >= ymin)
        mask &= (z <= zmax)&(z >= zmin) 
        xi = x[mask]
        yi = y[mask]
        zi = z[mask]
    
    xi = (((xi-xmin)/(xmax-xmin))*(nside-1)).astype(int)
    yi = (((yi-ymin)/(ymax-ymin))*(nside-1)).astype(int)
    zi = (((zi-zmin)/(zmax-zmin))*(nside-1)).astype(int)
    
    for xtup in zip(xi, yi, zi):
        grid[xtup] += 1
    
    close_struct = np.ones((closure, closure, closure), dtype=bool)
    
    mbgrid = grid >= count_threshold
    mbgrid = scipy.ndimage.morphology.binary_dilation(mbgrid, close_struct)
    mbgrid = scipy.ndimage.morphology.binary_erosion(mbgrid, close_struct)
    
    vertices, facets = skimage.measure.marching_cubes(mbgrid, 0.5)
    marching_cube_mesh = stl.mesh.Mesh(np.zeros(facets.shape[0], dtype=stl.mesh.Mesh.dtype))
    for i, f in enumerate(facets):
        for j in range(3):
            marching_cube_mesh.vectors[i][j] = vertices[f[j],:]
    
    return marching_cube_mesh

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("cloud")
    parser.add_argument("--output", required=True)
    parser.add_argument("--grid-size", default = 70)
    parser.add_argument("--opacity-cut", default = 0.05, type=float)
    parser.add_argument("--edge-behavior", default="clip")
    parser.add_argument("--closure", default=3, type=int)
    
    args = parser.parse_args()
    
    cloud = pd.read_csv(args.cloud)
    cloud = cloud[cloud["opacity"] > np.percentile(cloud["opacity"], 100*args.opacity_cut)]
    
    mesh = cloud2mesh(
        cloud,
        nside=args.grid_size,
        edge_behavior=args.edge_behavior,
    )
    mesh.save(args.output)
