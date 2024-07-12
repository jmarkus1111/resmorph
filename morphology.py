''' Functions to analyze the morphologies of galaxies by fitting Sersic profiles to photometry data.

    galight_free should be executed with a stacked image and PSF before executing galight_prior on multiple individual images.
'''

import os, glob, corner, sys
import numpy as np
import pickle as pkl
import astropy.coordinates
from astropy.io import fits
import webbpsf
import subprocess
from galight.data_process_Danial import DataProcess
from galight.fitting_specify_Danial import FittingSpecify
from galight.fitting_process_Danial import FittingProcess
os.environ['WEBBPSF_PATH'] = '/Users/jmark/OneDrive/Desktop/DARK REU/morphology_fitting/webbpsf-data'

results_dir = f'/Users/jmark/OneDrive/Desktop/DARK REU/morphology_fitting/morph/results'
if not os.path.exists(results_dir): os.mkdir(results_dir)
support_dir = '/Users/jmark/OneDrive/Desktop/DARK REU/morphology_fitting/support'

sys.path.insert(0, support_dir)

def make_PSFs(psf_dir, pixelscales=[0.02, 0.04], use_NIRCam=True, use_MIRI=False):
    ''' Generates PSFs for all NIRCam and/or MIRI filters and saves them to a directory.

    Parameters: 
    -----------
        psf_dir: str
            File path to where the outputted .fits PSF files are saved to.
        pixelscales: list of float
            All pixel scales to generate PSFs for.
        PSF_fine_sampling: int

        use_NIRCam: bool
            If PSFs for JWST NIRCam filters should be generated.
        use_MIRI: bool
            If PSFs for JWST MIRI filters should be generated.
    '''

    # import photometry
    NIRCam_SW_filters = ['f090w', 'f115w', 'f150w', 'f182m', 'f200w', 'f210m']
    NIRCam_LW_filters = ['f277w', 'f335m', 'f356w', 'f410m', 'f430m', 'f444w', 'f460m', 'f480m']
    MIRI_filters = ['f560w', 'f770w', 'f1000w', 'f1130w', 'f1280w', 'f1500w', 'f1800w', 'f2100w', 'f2550w']
        
    PSF_fine_sampling=1

    # make the PSF directory
    if not os.path.exists(psf_dir): os.mkdir(psf_dir)

    # merge the list of filters
    if use_NIRCam:
        PSF_filter_list = np.concatenate((NIRCam_SW_filters, NIRCam_LW_filters))
    if use_MIRI:
        PSF_filter_list = MIRI_filters
    if use_NIRCam and use_MIRI:
        PSF_filter_list = np.concatenate((NIRCam_SW_filters, NIRCam_LW_filters, MIRI_filters))

    # simulate the PSFs
    for pixelscale in pixelscales:
        for filter in PSF_filter_list:

            # NIRCam: 
            if use_NIRCam:
                nc = webbpsf.NIRCam()
                nc.filter = filter

                nc.pixelscale = pixelscale

                psf = nc.calc_psf(fov_pixels=99, oversample=PSF_fine_sampling)

            # MIRI: 
            if use_MIRI:
                mi = webbpsf.MIRI()
                mi.filter = filter

                mi.pixelscale = pixelscale

                psf = mi.calc_psf(fov_pixels=99, oversample=PSF_fine_sampling)

            psf = fits.PrimaryHDU(psf['DET_SAMP'].data)
            psf = fits.HDUList([psf])
            psf.writeto(f'{psf_dir}/{filter}_{pixelscale}.fits')


def galight_free(object, filter, save_dir, nsigma, npixels):
    ''' Fits sersic profile to object with a stacked image and stacked PSF.

    Parameters: 
    -----------
        object: str 
            The program ID and object ID of the object being fitted in the form f'{program_ID}_{object_ID}'.
        filter: str
            Filter for the PSF used. Should be 'SW' for the stacked PSF.
        save_dir: str
            File path to where the outputed stacked .fits files, .pdf plots, and .pkl dictionaries are saved to for all programs. The 
            stacked images and PSFs used in this function are stored here in their object folder.
        nsigma: float
            The s/n defined to detect all the objects in the image stamp.
        npixels: int
            The number of connected pixels, each greater than threshold, that an object must have to be detected.
    
    Returns:
    --------
        galight_dict: dict
            Contains Sersic and other paramters of the fit.
    '''

    # config
    program_ID = object.split('_')[0]
    object_ID = object.split('_')[1]
    save_path = f'{save_dir}/{program_ID}/{object_ID}'

    pixelscale = 0.04

    # read the data

    # reading the fits_files
    psf_file = f'{save_path}/{filter}_{pixelscale}.fits'
    PSF_data = fits.open(psf_file)[0].data
    
    sci_file = glob.glob(f'{save_path}/*{object_ID}*sci*')[0]
    sci_data = fits.open(sci_file)[0].data

    wht_file = glob.glob(f'{save_path}/*{object_ID}*wht*')[0]
    wht_data = fits.open(wht_file)[0].data
    err_data = np.power(wht_data, -0.5)

    # calculate the photometric zero point
    sci_header = fits.open(sci_file)[0].header
    zeropoint = 28.9
    # zeropoint = -2.5 * np.log10(sci_header['PHOTFLAM']) - 5 * np.log10(sci_header['PHOTPLAM']) - 2.408
    

    # the main target is at the centre of the FOV
    x_target, y_target = int(sci_data.shape[0]/2), int(sci_data.shape[1]/2)

    # run GALIGHT

    # GALIGHT pre-processing step
    data_process = DataProcess(fov_image=sci_data, fov_noise_map=err_data, header=sci_header,
                               target_pos=[x_target, y_target], pos_type='pixel',
                               rm_bkglight=True, if_plot=False, zp=zeropoint)

    radius = 1.2/pixelscale # arcsec/arcsec
    data_process.generate_target_materials(radius=radius, create_mask=False, if_plot=False,
                                           nsigma=nsigma, exp_sz=1.5, npixels=npixels,
                                           detect=True, detection_path=save_path)

    # add the PSF
    data_process.PSF_list = [PSF_data]

    # check if everything is there before attempting the run
    data_process.checkout()

    # sett up the model
    fit_sepc = FittingSpecify(data_process)
    fit_sepc.prepare_fitting_seq(point_source_num = 0)
    fit_sepc.build_fitting_seq()

    # set up the fitting method and run
    savename = f'{save_path}/galight_{filter}'
    fit_run = FittingProcess(fit_sepc, savename=savename, fitting_level=['shallow', 'deep'])
    fit_run.run(algorithm_list = ['MCMC', 'MCMC'])

    # save the results

    # save the plots
    fit_run.model_plot(save_plot=True, show_plot=False)
    fit_run.plot_final_galaxy_fit(target_ID=object, save_plot=True, show_plot=False)

    # change the name of the saved plot
    src = f'{save_path}/galight_{filter}_galaxy_final_plot.pdf'
    dst = f'{save_path}/galight_{filter}_galaxy_mix.pdf'
    subprocess.call(['mv', src, dst], shell=True)

    # save the output dict
    fit_run.dump_result()

    # make the visual inspection plot
    object_visual_inspection_path = f'{save_dir}/visual_inspection/{object}_{filter}.pdf'
    p = subprocess.Popen(f'cp {save_path}/galight_{filter}_model.pdf {object_visual_inspection_path}', stdout=subprocess.PIPE, shell=True)
    print(p.communicate())

    # read the best-fit parameters from the MCMC chain
    MCMC_headers = fit_run.chain_list[1][2]
    MCMC_chain   = fit_run.chain_list[1][1]

    MCMC_length  = int(MCMC_chain.shape[0])
    MCMC_weights = fit_run.chain_list[1][-1]

    galight_dict = {}
    galight_dict['corner'] = {}
    for i in range(len(MCMC_headers)):
        header = MCMC_headers[i]
        galight_dict['corner'][header] = corner.quantile(MCMC_chain[-MCMC_length:,i], q=[0.16, 0.50, 0.84], weights=MCMC_weights[-MCMC_length:])

    MCMC_flux = fit_run.mcmc_flux_list[-MCMC_length:]
    MCMC_mag = -2.5*np.log10(MCMC_flux) + zeropoint
    galight_dict['corner']['mag'] = np.quantile(MCMC_mag, q=[0.16, 0.50, 0.84])

    # galight automatic output
    galight_dict['galight_output'] = fit_run.final_result_galaxy

    # save the configs
    galight_dict['config'] = {}
    galight_dict['config']['object'] = object
    galight_dict['config']['filter'] = filter
    galight_dict['config']['pixelscale'] = pixelscale

    # save the dictionary
    with open(f'{results_dir}/{object}_{filter}.pkl', 'wb') as handle:
        pkl.dump(galight_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return galight_dict


def galight_prior(object, data_dir, psf_dir, save_dir, filter, nsigma, npixels):
    ''' Fits an object using some fixed sersic paramters from galight_free(). Should be ran on individual fiter images and their 
    corresponding PSFs.

    Parameters: 
    -----------
        object: str 
            The program ID and object ID of the object being fitted in the form f'{program_ID}_{object_ID}'.
        data_dir: str
            File path to where NIRCam and MIRI data are stored for all programs.
        psf_dir: str
            File path to where the .fits PSF files for filters are stored.
        save_dir: str
            File path to where the outputed stacked .fits files, .pdf plots, and .pkl dictionaries are saved to for all programs.
        filter: str
            Filter of the .fits sci file for the object to fit.
        nsigma: float
            The s/n defined to detect all the objects in the image stamp.
        npixels: int
            The number of connected pixels, each greater than threshold, that an object must
            have to be detected.
    
    Returns:
    --------
        galight_dict: dict
            Contains Sersic and other paramters of the fit.
    '''

    # config
    program_ID = int(object.split('_')[0])
    object_ID  = object.split('_')[1]
    object_dir = f'{data_dir}/{program_ID}/{object_ID}'
    save_path = f'{save_dir}/{program_ID}/{object_ID}'

    pixelscale = 0.04

    # read the data

    # read the fits_files
    PSF_path  = f'{psf_dir}/{filter}_{pixelscale}.fits'
    PSF_data = fits.open(PSF_path)[0].data
    
    sci_file = glob.glob(f'{object_dir}/*{filter}_*sci*')[0]
    sci_data = fits.open(sci_file)[0].data


    wht_file = glob.glob(f'{object_dir}/*{filter}_*wht*')[0]
    wht_data = fits.open(wht_file)[0].data
    err_data = np.power(wht_data, -0.5)

    # calculate the photometric zero point
    sci_header = fits.open(sci_file)[0].header
    zeropoint = 28.9
    # zeropoint = -2.5 * np.log10(sci_header['PHOTFLAM']) - 5 * np.log10(sci_header['PHOTPLAM']) - 2.408

    # the main target is at the centre of the FOV
    x_target, y_target = int(sci_data.shape[0]/2), int(sci_data.shape[1]/2)

    # run GALIGHT

    # GALIGHT pre-processing step
    data_process = DataProcess(fov_image=sci_data, fov_noise_map=err_data, header=sci_header,
                               target_pos=[x_target, y_target], pos_type='pixel',
                               rm_bkglight=True, if_plot=False, zp=zeropoint)

    radius = 1/pixelscale # arcsec/arcsec
    data_process.generate_target_materials(radius=radius, create_mask=False, if_plot=False,
                                           nsigma=nsigma, exp_sz=1.5, npixels=npixels,
                                           detect=False, detection_path=save_path)

    # add the PSF
    data_process.PSF_list = [PSF_data]

    # check if everything is there before attempting the run
    data_process.checkout()

    # set up the model
    fit_sepc = FittingSpecify(data_process)
    fit_sepc.prepare_fitting_seq(point_source_num = 0)
    fit_sepc.build_fitting_seq()

    # set up the prior (init, sigma, fixed, lower, upper)

    with open(f'{results_dir}/{object}_SW.pkl', 'rb') as handle:
        SW_galight_dict = pkl.load(handle)

    # init
    init_array = []
    for i in range(len(fit_sepc.kwargs_params['lens_light_model'][0])):

        temp_dict = {'R_sersic': SW_galight_dict['corner'][f'R_sersic_lens_light{i}'][1],
                     'n_sersic': SW_galight_dict['corner'][f'n_sersic_lens_light{i}'][1],
                     'e1': SW_galight_dict['corner'][f'e1_lens_light{i}'][1],
                     'e2': SW_galight_dict['corner'][f'e2_lens_light{i}'][1],
                     'center_x': SW_galight_dict['corner'][f'center_x_lens_light{i}'][1],
                     'center_y': SW_galight_dict['corner'][f'center_y_lens_light{i}'][1]}

        init_array.append(temp_dict)

    fit_sepc.kwargs_params['lens_light_model'][0] = init_array

    # sigma
    sigma_array = []
    for i in range(len(fit_sepc.kwargs_params['lens_light_model'][0])):

        temp_dict = {'R_sersic': max(np.diff(SW_galight_dict['corner'][f'R_sersic_lens_light{i}'])),
                     'n_sersic': max(np.diff(SW_galight_dict['corner'][f'n_sersic_lens_light{i}'])),
                     'e1': max(np.diff(SW_galight_dict['corner'][f'e1_lens_light{i}'])),
                     'e2': max(np.diff(SW_galight_dict['corner'][f'e2_lens_light{i}'])),
                     'center_x': max(np.diff(SW_galight_dict['corner'][f'center_x_lens_light{i}'])),
                     'center_y': max(np.diff(SW_galight_dict['corner'][f'center_y_lens_light{i}']))}
        print(max(np.diff(SW_galight_dict['corner'][f'R_sersic_lens_light{i}'])))

        sigma_array.append(temp_dict)

    fit_sepc.kwargs_params['lens_light_model'][1] = sigma_array

    # fixed
    fixed_array = []
    for i in range(len(fit_sepc.kwargs_params['lens_light_model'][0])):

        fixed_array.append({})

    fit_sepc.kwargs_params['lens_light_model'][2] = fixed_array

    # lower
    lower_array = []
    for i in range(len(fit_sepc.kwargs_params['lens_light_model'][0])):

        R_min = fit_sepc.kwargs_params['lens_light_model'][3][i]['R_sersic']

        temp_dict = {'R_sersic': R_min,
                     'n_sersic': SW_galight_dict['corner'][f'n_sersic_lens_light{i}'][0],
                     'e1': SW_galight_dict['corner'][f'e1_lens_light{i}'][0],
                     'e2': SW_galight_dict['corner'][f'e2_lens_light{i}'][0],
                     'center_x': SW_galight_dict['corner'][f'center_x_lens_light{i}'][0],
                     'center_y': SW_galight_dict['corner'][f'center_y_lens_light{i}'][0]}

        lower_array.append(temp_dict)

    fit_sepc.kwargs_params['lens_light_model'][3] = lower_array

    # upper
    upper_array = []
    for i in range(len(fit_sepc.kwargs_params['lens_light_model'][0])):

        R_max = fit_sepc.kwargs_params['lens_light_model'][4][i]['R_sersic']

        temp_dict = {'R_sersic': R_max,
                     'n_sersic': SW_galight_dict['corner'][f'n_sersic_lens_light{i}'][2],
                     'e1': SW_galight_dict['corner'][f'e1_lens_light{i}'][2],
                     'e2': SW_galight_dict['corner'][f'e2_lens_light{i}'][2],
                     'center_x': SW_galight_dict['corner'][f'center_x_lens_light{i}'][2],
                     'center_y': SW_galight_dict['corner'][f'center_y_lens_light{i}'][2]}

        upper_array.append(temp_dict)

    fit_sepc.kwargs_params['lens_light_model'][4] = upper_array

    # set up the fitting method and running
    savename = f'{save_path}/galight_{filter}'
    fit_run = FittingProcess(fit_sepc, savename=savename, fitting_level=['shallow', 'deep'])
    fit_run.run(algorithm_list = ['MCMC', 'MCMC'])

    # save the results

    # save the plots
    fit_run.model_plot(save_plot=True, show_plot=False)
    fit_run.plot_final_galaxy_fit(target_ID=object, save_plot=True, show_plot=False)

    # change the name of the saved plot
    src = f'{save_path}/galight_{filter}_galaxy_final_plot.pdf'
    dst = f'{save_path}/galight_{filter}_galaxy_mix.pdf'
    subprocess.call(['mv', src, dst], shell=True)

    # save the output dict
    fit_run.dump_result()

    # make the visual inspection plot
    object_visual_inspection_path = f'{save_dir}/visual_inspection/{object}_{filter}.pdf'
    p = subprocess.Popen(f'cp {save_path}/galight_{filter}_model.pdf {object_visual_inspection_path}', stdout=subprocess.PIPE, shell=True)
    print(p.communicate())

    # read the best-fit parameters from the MCMC chain
    MCMC_headers = fit_run.chain_list[1][2]
    MCMC_chain   = fit_run.chain_list[1][1]

    MCMC_length  = int(MCMC_chain.shape[0])
    MCMC_weights = fit_run.chain_list[1][-1]

    galight_dict = {}
    galight_dict['corner'] = {}
    for i in range(len(MCMC_headers)):
        header = MCMC_headers[i]
        galight_dict['corner'][header] = corner.quantile(MCMC_chain[-MCMC_length:,i], q=[0.16, 0.50, 0.84], weights=MCMC_weights[-MCMC_length:])

    MCMC_flux = fit_run.mcmc_flux_list[-MCMC_length:]
    MCMC_mag = -2.5*np.log10(MCMC_flux) + zeropoint
    galight_dict['corner']['mag'] = np.quantile(MCMC_mag, q=[0.16, 0.50, 0.84])

    # galight automatic output
    galight_dict['galight_output'] = fit_run.final_result_galaxy

    # save the configs
    galight_dict['config'] = {}
    galight_dict['config']['object'] = object
    galight_dict['config']['filter'] = filter
    galight_dict['config']['pixelscale'] = pixelscale

    # save the dictionary
    with open(f'{results_dir}/{object}_{filter}.pkl', 'wb') as handle:
        pkl.dump(galight_dict, handle, protocol=pkl.HIGHEST_PROTOCOL)

    return galight_dict
