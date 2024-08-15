''' Functions related to the Sersic half light radius.
'''

import pickle as pkl
from astropy.cosmology import FlatLambdaCDM
import numpy as np
import matplotlib.pyplot as plt
import glob
from fitting_pipeline import get_z

def get_R_sersic(results_dir, object, z):
    ''' Helper function that gets the effective radius from the Sersic profile fitted to a given object.

    Parameters: 
    -----------
        results_dir: str
            File pah to where the .pkl dicts containing the fitted Sersic parameters are saved.
        object: str 
            The program ID and object ID of the object in the form f'{program_ID}_{object_ID}'.
        z: float
            Redshift of the object.
    
    Returns:
    --------
        R_sersic_pc: float
            Effective radius (in parsecs) from the Sersic profile fit of the filter containing the redshifted 2000 angstrom wavelength.
    '''

    # set cosmology 
    cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.308)

    program_ID = object.split('_')[0]
    object_ID  = object.split('_')[1]

    # dict of all JWST NIRCam filters and their respective wavelength interval in angstroms
    NIRCam_filters = {'f070w': [6048.20, 7927.07], 'f090w': [7881.88, 10243.08], 'f115w': [9975.60, 13058.40], 'f140m': [13042.25, 15058.58], 
                      'f150w': [13041.19, 16948.89], 'f162m': [15126.16, 17439.17], 'f164n': [16171.41, 16717.72], 'f150w2': [9774.71, 23946.87], 
                      'f182m': [16959.53, 20010.97], 'f187n': [18445.28, 19029.98], 'f200w': [17249.08, 22596.64], 'f210m': [19618.54, 22337.29], 
                      'f212n': [20900.93, 21524.99], 'f250m': [23935.49, 26177.91], 'f277w': [23673.12, 32203.22], 'f300m': [27703.55, 32505.92], 
                      'f323n': [32046.29, 32761.07], 'f322w2': [23851.45, 41234.69], 'f335m': [31203.36, 36442.23], 'f356w': [30732.91, 40801.26], 
                      'f360m': [33260.34, 39037.39], 'f405n': [40097.87, 40966.10], 'f410m': [37763.56, 44048.41], 'f430m': [41227.68, 44448.79], 
                      'f444w': [38039.57, 50995.50], 'f460m': [44652.64, 48146.41], 'f466n': [46021.35, 47042.62], 'f470n': [46553.98, 47566.82], 
                      'f480m': [45820.02, 50919.02]}
    
    # get list of filters that include contain 2000 angstroms when corrected for the redshift of the object
    filters_including_2000A = []
    for filter in NIRCam_filters:
        min = NIRCam_filters[filter][0]
        max = NIRCam_filters[filter][1]

        # correct for redshift
        min /= (z + 1)
        max /= (z + 1)

        if 2000 > min and 2000 <= max:
            filters_including_2000A.append(filter)

    # get list of filters used with the given object
    dict_list = glob.glob(f'{results_dir}/{program_ID}_{object_ID}_*.pkl')
    filters_used = [dict.split('_')[-1][0:-4] for dict in dict_list]

    # find the common filters between the two lists 
    valid_filters_used = list(set(filters_including_2000A) & set(filters_used))
    if not valid_filters_used:
        print('NO FILTERS INCLUDE 2000 A')
        return 

    # get Sersic parameters from the first valid filter's dict 
    with open(f'{results_dir}/{program_ID}_{object_ID}_{valid_filters_used[0]}.pkl', 'rb') as handle:
        object_dict = pkl.load(handle)

    galight_output_dict = object_dict['galight_output'][0] # dict with all Sersic parameters
    R_sersic = galight_output_dict['R_sersic'] # effective radius in arcseconds (?)

    # convert to pc
    d_A = cosmo.angular_diameter_distance(z)
    theta_radian = R_sersic * np.pi / 180 / 3600
    R_sersic_pc = d_A * theta_radian * 1000000 # also converting from Mpc to pc
    
    R_sersic_pc = R_sersic_pc.value # remove astropy units
    return R_sersic_pc


def r_vs_wavelength(results_dir, data_dir):
    # set cosmology 
    cosmo = FlatLambdaCDM(H0 = 70, Om0 = 0.308)

    # dict of all JWST NIRCam filters and their respective wavelength interval in angstroms
    NIRCam_filters = {'f070w': [6048.20, 7927.07], 'f090w': [7881.88, 10243.08], 'f115w': [9975.60, 13058.40], 'f140m': [13042.25, 15058.58], 
                    'f150w': [13041.19, 16948.89], 'f162m': [15126.16, 17439.17], 'f164n': [16171.41, 16717.72], 'f150w2': [9774.71, 23946.87], 
                    'f182m': [16959.53, 20010.97], 'f187n': [18445.28, 19029.98], 'f200w': [17249.08, 22596.64], 'f210m': [19618.54, 22337.29], 
                    'f212n': [20900.93, 21524.99], 'f250m': [23935.49, 26177.91], 'f277w': [23673.12, 32203.22], 'f300m': [27703.55, 32505.92], 
                    'f323n': [32046.29, 32761.07], 'f322w2': [23851.45, 41234.69], 'f335m': [31203.36, 36442.23], 'f356w': [30732.91, 40801.26], 
                    'f360m': [33260.34, 39037.39], 'f405n': [40097.87, 40966.10], 'f410m': [37763.56, 44048.41], 'f430m': [41227.68, 44448.79], 
                    'f444w': [38039.57, 50995.50], 'f460m': [44652.64, 48146.41], 'f466n': [46021.35, 47042.62], 'f470n': [46553.98, 47566.82], 
                    'f480m': [45820.02, 50919.02]}

    dict_list = glob.glob(f'{results_dir}/*w.pkl')

    R_e = []
    wavelength = []
    for dict in dict_list:
        if 'SW' not in dict and 'LW' not in dict:
            # get z
            program_ID = dict.split('_')[-3].split('\\')[-1]
            object_ID = dict.split('_')[-2]
            object = f'{program_ID}_{object_ID}'
            z = get_z(object, data_dir)

            filter = dict.split('_')[-1][0:-4]
            filter_wavelengths  = NIRCam_filters[filter]
            wavelength_avg = np.average(filter_wavelengths)
            wavelength_avg /= (z + 1) # redshift wavelength
            wavelength.append(wavelength_avg)

            with open(dict, 'rb') as handle:
                object_dict = pkl.load(handle)
            galight_output_dict = object_dict['galight_output'][0] # dict with all Sersic parameters
            R_sersic = galight_output_dict['R_sersic'] # effective radius in arcseconds (?)
            
            # convert to pc
            d_A = cosmo.angular_diameter_distance(z)
            theta_radian = R_sersic * np.pi / 180 / 3600
            R_sersic_pc = d_A * theta_radian * 1000000 # also converting from Mpc to pc
            
            R_sersic_pc = R_sersic_pc.value # remove astropy units
            R_e.append(R_sersic_pc)

    plt.scatter(R_e, wavelength)
    plt.ylabel('R [pc]')
    plt.xlabel('Filter Average Rest Frame Wavelength [A]')
