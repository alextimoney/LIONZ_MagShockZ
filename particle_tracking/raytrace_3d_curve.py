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
    parser.add_argument('-f', '--flash_file', type=str, default='/scratch/ckuranz_root/ckuranz1/timoney/MagShockZ/simuls/3d_shield_curved/MagShockZ_hdf5_plt_cnt_0019')
    parser.add_argument('-s', '--scaling_factor', type=float, default=1.0, help="Artificial scaling factor for electron density")
    parser.add_argument('-o', '--output_file', type=str, default=f"", help="additional string to add to output")
    
    args = parser.parse_args()
    logger.info(args)
    # Save metadata
    metadata = {
        'num_photons': int(args.num_photons),
        'beam_size': args.beam_size,
        'divergence': args.divergence,
        'flash_file': args.flash_file,
        'scaling_factor': args.scaling_factor,
        # 'output_file': args.output_file,
        'num_processors': mp.cpu_count(),
        'timestamp': time.time()
    }

    Np=int(args.num_photons)
    beam_size = args.beam_size
    divergence = args.divergence
    scaling_factor = args.scaling_factor
    output_file = args.output_file

    ds = yt.load(args.flash_file)

    def make_electron_number_density(field, data):
        N_A = yt.units.yt_array.YTQuantity(6.02214076e23, "1/mol")
        proton_mass = yt.units.yt_array.YTQuantity(1.6726219e-24, 'g')
        electron_number_density = N_A*data["flash","dens"]*data["flash","ye"]/proton_mass
        return electron_number_density
    ds.add_field(("flash", "edens"), function=make_electron_number_density, units="1/code_length**3",sampling_type="cell") # same here

    level = 0
    dims = ds.domain_dimensions * ds.refine_by**level
    all_data = ds.covering_grid(
        level,
        left_edge=ds.domain_left_edge,
        dims=dims,
    )

    start_time = time.perf_counter()

    num_processors = mp.cpu_count() // 2

    # This assumes FLASH data is in cgs - converts to m
    # using regular ray tracing 
    x_coords = all_data[('flash','x')][:,0,0].value*1e-2
    y_coords = all_data[('flash','y')][0,:,0].value*1e-2
    z_coords = all_data[('flash','z')][0,0,:].value*1e-2
    # electron_density = all_data[("flash", "edens")].value * 1e6 * scaling_factor
    # electron_density = electron_density * np.mean(electron_density) # reduce background to see what happens
    y_coords -= 0.008

    XX, YY, ZZ = np.meshgrid(x_coords, y_coords, z_coords, indexing='ij')

    # Find midplane index in z (probing direction)
    mid_z = z_coords[len(z_coords)//2]
    print(f"Midplane z-coordinate: {mid_z:.16f} m")

    # define critical density and bg dens
    nc = 1.0e27  # critical density for 1053 nm in m^-3
    bg_density = 3e24  # uniform background dens

    sphere_radius = 4.0e-3  # 4.0 mm radius
    x0 = 0.0
    y0 = 0.0
    z0 = mid_z  # Centers the sphere in the middle of the probing axis

    # choose a peak density safely below critical density to prevent severe clipping
    peak_density = 0.7 * nc  

    # calculate the 3D radial distance from the center for every grid cell        
    RR_3D = np.sqrt((XX - x0)**2 + (YY - y0)**2 + (ZZ - z0)**2)

    # define the spatial gradient profile
    r_norm = RR_3D / sphere_radius
    # linear profile
    radial_profile = 1.0 - r_norm
    # print('radial profile', radial_profile)

    # calculate the linear density gradient
    linear_gradient = bg_density + (peak_density - bg_density) * radial_profile

    # if inside the sphere, use the gradient. if outside, use the uniform background.
    electron_density = np.where(((RR_3D < sphere_radius)), linear_gradient, bg_density)

    logger.info(f"Gradient sphere injected at ({x0}, {y0}, {z0}) m")
    logger.info(f"Sphere Radius: {sphere_radius*1e3:.1f} mm | Peak Density: {peak_density:.2e} m^-3")

    # ---------------------------------
    
    # print(electron_density, 'electron_density shape:', electron_density.shape)
    # electron_density = electron_density/1e22 #scale down
    # print(electron_density)

    print('x_coords shape:', x_coords.shape)   # (Nx,)
    print('y_coords shape:', y_coords.shape)   # (Ny,)
    print('z_coords shape:', z_coords.shape)   # (Nz,)
    print('electron_density shape:', electron_density.shape)   # (Nx, Ny, Nz)

    # y adjustment. Tune this
    # y_coords -= 0.004

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

    ID = metadata['flash_file'][-4:]  # Get plot number from filename for easy identification
    # output_dir = f"/home/timoney/scratch/timoney/MagShockZ/traces/3d_noshield_scaledens/raytrace_{ID}_{output_file}"
    output_dir = f"/home/timoney/scratch/timoney/MagShockZ/LIONZ_MagShockZ/outputs/traces/synthetic_trace/curve_v0"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(os.path.join(output_dir, f'ray_output.npy'),'wb') as f:
        np.save(f, output)

    with open(os.path.join(output_dir, 'metadata.txt'),'w') as f:
        for key, value in metadata.items():
            f.write(f"{key}: {value}\n")

    with open(os.path.join(output_dir, 'density.npy'),'wb') as f:
        np.save(f, electron_density)
