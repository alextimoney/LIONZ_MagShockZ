"""
Run Jack Hare's synthetic shadowgraphy code with multiprocessing
Splits up rays between multiple processes.
"""

import numpy as np
import multiprocessing as mp
import yt
import time
import particle_tracker as pt
import os
import logging

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
    parser.add_argument('-f', '--flash_file', type=str, default='/gpfs/accounts/ckuranz_root/ckuranz1/timoney/MagShockZ_LIONZ_timoney/simuls/flat_hdf5_plt_cnt_0345')
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

    # downsample     

    # downsize data for testing
    x_min, x_max = -1, 1  # in cm
    y_min, y_max = -1, 1  # in cm

    print(f"x range: {x_min} to {x_max} cm")
    print(f"y range: {y_min} to {y_max} cm")

    # define your target resolution : original is 3008 x 2112
    nx_new = 752 
    ny_new = 528


    slc = ds.slice(2, 0) # Slice perpendicular to Z (axis 2) at the center
    # frb = slc.to_frb(
    #     width=(x_max-x_min, y_max-y_min), 
    #     resolution=(nx_new, ny_new), 
    #     center=((x_min+x_max)/2, (y_min+y_max)/2, 0)
    # )

    # create a slice and use an FRB to downsample
    # slc = ds.slice(2, 0) # Slice perpendicular to Z (axis 2) at the center
    frb = slc.to_frb((x_max - x_min), (nx_new, ny_new), center=((x_min+x_max)/2, (y_min+y_max)/2, 0))

    dx = ds.domain_width / ds.domain_dimensions

    new_dims = np.array([
        int((x_max - x_min) / dx[0]),
        int((y_max - y_min) / dx[1]),
        ds.domain_dimensions[2] # Keep original Z resolution for now
    ])

    level = 0
    dims = ds.domain_dimensions * ds.refine_by**level
    all_data = ds.covering_grid(
        level,
        left_edge=ds.domain_left_edge,
        dims=new_dims,
    )

    start_time = time.perf_counter()

    num_processors = mp.cpu_count() // 2

    # This assumes FLASH data is in cgs - converts to m
    # x_coords = all_data[('flash','x')][:,0,0].value*1e-2
    # y_coords = all_data[('flash','y')][0,:,0].value*1e-2
    # downsampled coords
    x_coords = np.linspace(x_min, x_max, nx_new) * 1e-2 # to meters
    y_coords = np.linspace(y_min, y_max, ny_new) * 1e-2 # to meters

    # set up z coords (in m)
    z_min = -5e-3      # = -width [m] * slices/2          
    z_max = 5e-3       # = width [m] * slices/2
    Nz = 505             # number of slices
    # Uniformly spaced z coordinates
    z_coords = np.linspace(z_min, z_max, Nz)

    electron_density_2d = frb[("flash", "edens")].value * 1e6 * scaling_factor # downsampled edens in 1/m^3
    electron_density_2d = np.transpose(electron_density_2d)  # Transpose to match x,y orientation

    # electron_density_2d = all_data[('flash','edens')].value*1e6*scaling_factor
    electron_density_2d = np.squeeze(electron_density_2d)
    electron_density_3d = np.repeat(electron_density_2d[:, :, np.newaxis], Nz, axis=2)
    electron_density_3d = electron_density_3d.astype(np.float32)

    print('x_coords shape:', x_coords.shape)   # (Nx,)
    print('y_coords shape:', y_coords.shape)   # (Ny,)
    print('z_coords shape:', z_coords.shape)   # (Nz,)
    print('electron_density_3d shape:', electron_density_3d.shape)   # (Nx, Ny, Nz)

    # y adjustment. Tune this
    y_coords -= 0.008

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

    output_dir = f"/home/timoney/scratch/timoney/MagShockZ/traces/raytrace_2d"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f'ray_output.npy'),'wb') as f:
        np.save(f, output)

    with open(os.path.join(output_dir, 'metadata.txt'),'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")
