"""
Run Jack Hare's synthetic shadowgraphy code with multiprocessing
Splits up rays between multiple processes.
"""

import numpy as np
import multiprocessing as mp
import yt
import time
import particle_tracker_ke as pt
import os
import logging
from scipy.ndimage import map_coordinates
import dimensionalize as dim
from scipy.ndimage import gaussian_filter

logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def solve_beam(args):
    proc_id, x_coords, y_coords, z_coords, electron_density_3d, Np_per_proc, beam_size, divergence = args
    
    #random seed
    np.random.seed(proc_id + int(time.time()) % 100000)

    # Create new ElectronCube for each process
    cube = pt.ElectronCube(x_coords, y_coords, z_coords)
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
    parser.add_argument('-f', '--flash_file', type=str, default='/home/timoney/scratch/timoney/Geometries/FLAT/MagShockZ_hdf5_plt_cnt_0100')
    parser.add_argument('-s', '--scaling_factor', type=float, default=1.0, help="Artificial scaling factor for electron density")
    
    args = parser.parse_args()
    logger.info(args)
    # Save metadata
    metadata = {
        'num_photons': int(args.num_photons),
        'beam_size': args.beam_size,
        'divergence': args.divergence,
        'flash_file': args.flash_file,
        'scaling_factor': args.scaling_factor,
        'num_processors': mp.cpu_count(),
        'timestamp': time.time()
    }

    Np=int(args.num_photons)
    beam_size = args.beam_size
    divergence = args.divergence
    scaling_factor = args.scaling_factor

    ds = yt.load(args.flash_file)

    def make_electron_number_density(field, data):
        N_A = yt.units.yt_array.YTQuantity(6.02214076e23, "1/mol")
        proton_mass = yt.units.yt_array.YTQuantity(1.6726219e-24, 'g')
        electron_number_density = N_A*data["flash","dens"]*data["flash","ye"]/proton_mass
        return electron_number_density
    ds.add_field(("flash", "edens"), function=make_electron_number_density, units="1/code_length**3",sampling_type="cell") # same here

    # (x,y,z) = (384,496,384)

    # define domain size
    x_min, x_max = -0.8, 0.8  # in cm
    y_min, y_max = -0.075, 2.0  # in cm
    z_min, z_max = -0.8, 0.8  # in cm

    print(f"x range: {x_min} to {x_max} cm")
    print(f"y range: {y_min} to {y_max} cm")
    print(f"z range: {z_min} to {z_max} cm")

    # define dimensions
    Nx = 384
    Ny = 496
    Nz = 384

    density_3d = np.zeros((Nx, Ny, Nz))

    # create band
    # x_center = Nx // 2
    # y_center = Ny // 2

    # band_width_x = 200
    # band_width_y = 50 

    # x_start = x_center - (band_width_x // 2)
    # x_end = x_center + (band_width_x // 2)
    # y_start = y_center - (band_width_y // 2)
    # y_end = y_center + (band_width_y // 2)
    # z_start = -50
    # z_end = 50

    # density_3d[x_start:x_end, y_start:y_end, z_start:z_end] = 1
    # smooth_density_3d = gaussian_filter(density_3d, sigma=30.0)

    start_time = time.perf_counter()

    num_processors = mp.cpu_count() // 2

    # This assumes FLASH data is in cgs - converts to m
    # x_coords = all_data[('flash','x')][:,0,0].value*1e-2
    # y_coords = all_data[('flash','y')][0,:,0].value*1e-2
    # downsampled coords
    x_coords = np.linspace(x_min, x_max, Nx) * 1e-2 # to meters
    y_coords = np.linspace(y_min, y_max, Ny) * 1e-2 # to meters
    z_coords = np.linspace(z_min, z_max, Nz) * 1e-2 # to meters

    electron_density_3d = density_3d
    # electron_density_3d = smooth_density_3d * 1e6 * scaling_factor  # Convert from cm^-3 to m^-3 and apply scaling factor

    print('x_coords shape:', x_coords.shape)   # (Nx,)
    print('y_coords shape:', y_coords.shape)   # (Ny,)
    print('z_coords shape:', z_coords.shape)   # (Nz,)
    print('electron_density_3d shape:', electron_density_3d.shape)   # (Nx, Ny, Nz)

    # y adjustment. Tune this
    y_coords -= 0.004

    Np_per_proc = Np // num_processors
    logger.info(f"Number of photons per processor: {Np_per_proc}")
    
    process_args = [(i, x_coords, y_coords, z_coords, electron_density_3d, Np_per_proc, beam_size, divergence)
                    for i in range(num_processors)]

    with mp.Pool(num_processors) as p:
        output = p.map(solve_beam, process_args)

    output = np.concatenate(output, axis=1)

    print(output.shape)

    end_time = time.perf_counter()
    logger.info("Ray tracing completed.")

    logger.info(f"Time taken: {end_time - start_time:.2f} seconds for {Np} rays")
    logger.info(f"Average time per ray: {(end_time - start_time) / Np:.6f} seconds")

    ID = metadata['flash_file'][-4:]  # Get plot number from filename for easy identification
    output_dir = f"/home/timoney/scratch/timoney/MagShockZ/traces/synthetic_trace/raytrace_{ID}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f'ray_output.npy'),'wb') as f:
        np.save(f, output)

    with open(os.path.join(output_dir, 'metadata.txt'),'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
