import numpy as np
from scipy.ndimage import map_coordinates

#make 2d ED 3d by rotating
def rotational_2d_to_3d(electron_density_2d, Nz, z_coords):
    """
    Revolves a 2D array around its central Y-axis, treating the 
    left and right halves as separate cylindrically symmetric domains.
    """
    # 1. get dimensions
    Ny, Nx = electron_density_2d.shape
    x_center = Nx / 2.0  
    z_center = Nz / 2.0
    
    # 3. create 3D coordinate grid
    # x_idx, y_idx, z_idx = np.mgrid[0:Nx, 0:Ny, 0:Nz]
    y_idx, x_idx, z_idx = np.mgrid[0:Ny, 0:Nx, 0:Nz]
    
    # 4. calculate distance from the center for X and Z
    dx = x_idx - x_center
    dz = z_idx - z_center
    
    # calculate the 3D radial distance from the Y-axis
    r_idx = np.sqrt(dx**2 + dz**2)
    
    # 5. THE SPLIT MAPPING: 
    # if dx >= 0 (right side of 3D volume), map to right side of 2D array
    # if dx < 0 (left side of 3D volume), map to left side of 2D array
    lookup_x = np.where(dx >= 0, x_center + r_idx, x_center - r_idx)
    
    # 6. prepare coordinates for interpolation
    coords = np.array([y_idx, lookup_x])
    
    # 7. interpolate to populate the 3D array
    electron_density_3d = map_coordinates(electron_density_2d, coords, order=1, mode='constant', cval=0.0)
    
    return electron_density_3d