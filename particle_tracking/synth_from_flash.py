"""
Run Jack Hare's synthetic shadowgraphy code with multiprocessing
Splits up rays between multiple processes.
"""

import numpy as np
import multiprocessing as mp
import yt
import time
import particle_tracker as pt
import ray_transfer_matrix as rtm
import os
import logging
import scipy.constants as sc

c = sc.c

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def solve_beam(args):
    proc_id, x_coords, y_coords, z_coords, electron_density_3d, Np_per_proc, beam_size, divergence = args
    
    #random seed
    np.random.seed(proc_id + int(time.time()) % 100000)

    # Create new ElectronCube for each process
    cube = pt.ElectronCube(x_coords, y_coords, z_coords, probing_direction = 'z')
    logger.info(f"Process {proc_id}: ElectronCube created with shape {electron_density_3d.shape}")
    cube.external_ne(electron_density_3d)

    cube.calc_dndr()
    cube.init_beam(Np=Np_per_proc, beam_size=beam_size, divergence=divergence)
    logger.info(f"Process {proc_id}: Beam initialized with {Np_per_proc} photons")

    logger.info(f"Process {proc_id}: Starting ray tracing")
    return cube.solve()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run parallel ray tracing.")
    parser.add_argument('-n', '--num_photons', type=int, default=1e6, help="Number of photons to ray trace")
    parser.add_argument('-b', '--beam_size', type=float, default=12e-3, help="Beam size in millimeters")
    parser.add_argument('-d', '--divergence', type=float, default=0, help="Divergence in milliradians")
    parser.add_argument('-o', '--output_file', type=str, default=f"", help="additional string to add to output")
    
    args = parser.parse_args()
    logger.info(args)
    # Save metadata
    metadata = {
        'num_photons': int(args.num_photons),
        'beam_size': args.beam_size,
        'divergence': args.divergence,
        # 'output_file': args.output_file,
        'num_processors': mp.cpu_count(),
        
        'timestamp': time.time()
    }

    Np=int(args.num_photons)
    beam_size = args.beam_size
    divergence = args.divergence
    output_file = args.output_file

    start_time = time.perf_counter()

    num_processors = mp.cpu_count() // 2
    
    half_box_m = 0.008 # 8 mm half box size
    n_grid = 101 # number of grid points in each dimension

    x_coords = np.linspace(-half_box_m, half_box_m, n_grid)
    y_coords = np.linspace(-half_box_m, half_box_m, n_grid)
    z_coords = np.linspace(-half_box_m, half_box_m, n_grid)

    XX, YY, ZZ = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # Find midplane indices
    mid_x = (x_coords.min() + x_coords.max()) / 2.0
    mid_y = (y_coords.min() + y_coords.max()) / 2.0
    mid_z = (z_coords.min() + z_coords.max()) / 2.0

    # define critical density and bg dens
    nc = 1.0e27  # critical density for 1053 nm in m^-3
    bg_density = 3e24  # uniform background dens

    sphere_radius = 4.0e-3  # 4.0 mm radius
    x0 = mid_x
    y0 = mid_y
    z0 = mid_z  # Centers the sphere in the middle of the probing axis
    print(((z_coords.max() - z_coords.min()) / 2) * 1000) # 8.0 mm

    # choose a peak density safely below critical density to prevent severe clipping
    peak_density = 0.7 * nc  

    # make sphere     
    RR_3D = np.sqrt((XX - x0)**2 + (YY - y0)**2 + (ZZ - z0)**2)
    r_norm = RR_3D / sphere_radius
    radial_profile = 1.0 - r_norm
    linear_gradient = bg_density + (peak_density - bg_density) * radial_profile
    electron_density = np.where(((RR_3D < sphere_radius)), linear_gradient, bg_density)

    logger.info(f"Gradient sphere injected at ({x0}, {y0}, {z0}) m")
    logger.info(f"Sphere Radius: {sphere_radius*1e3:.1f} mm | Peak Density: {peak_density:.2e} m^-3")

    print('x_coords shape:', x_coords.shape)   # (Nx,)
    print('y_coords shape:', y_coords.shape)   # (Ny,)
    print('z_coords shape:', z_coords.shape)   # (Nz,)
    print('electron_density shape:', electron_density.shape)   # (Nx, Ny, Nz)

    Np_per_proc = Np // num_processors
    logger.info(f"Number of photons per processor: {Np_per_proc}")
    
    process_args = [(i, x_coords, y_coords, z_coords, electron_density, Np_per_proc, beam_size, divergence)
                    for i in range(num_processors)]

    with mp.Pool(num_processors) as p:
        output = p.map(solve_beam, process_args)

    output = np.concatenate(output, axis=1)
    print("Output shape:", output.shape)

    end_time = time.perf_counter()
    logger.info("Ray tracing completed.")

    logger.info(f"Time taken: {end_time - start_time:.2f} seconds for {Np} rays")
    logger.info(f"Average time per ray: {(end_time - start_time) / Np:.6f} seconds")

    output_dir = f"/home/timoney/scratch/timoney/MagShockZ/LIONZ_MagShockZ/outputs/traces/synthetic_trace/curve_sff_odd"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f'ray_output.npy'),'wb') as f:
        np.save(f, output)

    with open(os.path.join(output_dir, 'metadata.txt'),'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    with open(os.path.join(output_dir, 'density.npy'),'wb') as f:
        np.save(f, electron_density)