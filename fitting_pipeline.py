''' Functions that help automatically fit Sersic profiles to a large number of galaxies.

    Only fit_all_objects needs to be executed to fit everyting in a given directory. 

    fit_object can be used to fit a specific object for all filters.
'''

import pickle as pk
import glob
import os
import PyPDF2
from morphology import *
from photometry import *
from weighted_residual import *

def get_z(object, data_dir):
    ''' Returns the redshift of a given object from redshifts_dict.pk.

    Parameters: 
    -----------
        object: str 
            The program ID and object ID of the object in the form f'{program_ID}_{object_ID}'.
        data_dir: str
            File path to redshifts_dict.pkl.
    
    Returns:
    --------
        z: float
            Redshift of the object.
    '''

    program_ID = object.split('_')[0]
    object_ID  = object.split('_')[1]

    with open(f'{data_dir}/{program_ID}/redshifts_dict.pkl', 'rb') as handle:
        redshifts_dict = pk.load(handle)

    z = redshifts_dict.get(f'{program_ID}_{object_ID}')
    return z

 
def get_filter_list(object, data_dir, z):
    ''' Helper function that gets the list of filters that are NirCam SW, above the redhsifted LyAlpha break, and unconvolved.

    Parameters: 
    -----------
        object: str 
            The program ID and object ID of the object in the form f'{program_ID}_{object_ID}'.
        data_dir: str
            File path to where NIRCam and MIRI data are stored for all programs.
        z: float
            Redshift of the object.
    
    Returns:
    --------
        filter_list: list of str
            All filters that meet the requirements.
    '''

    program_ID = object.split('_')[0]
    object_ID  = object.split('_')[1]

    filter_list = []

    LyAlpha = 1215.67
    shifted_LyAlpha = LyAlpha * (z + 1)

    # dict of all JWST short wavelength filters and their respective wavelength interval in angstroms
    SW_filters = {'f070w': [6048.20, 7927.07], 'f090w': [7881.88, 10243.08], 'f115w': [9975.60, 13058.40], 'f140m': [13042.25, 15058.58], 
                'f150w': [13041.19, 16948.89], 'f162m': [15126.16, 17439.17], 'f164n': [16171.41, 16717.72], 
                'f150w2': [9774.71, 23946.87], 'f182m': [16959.53, 20010.97], 'f187n': [18445.28, 19029.98], 
                'f200w': [17249.08, 22596.64], 'f210m': [19618.54, 22337.29], 'f212n': [20900.93, 21524.99]}

    # get list of filtered .sci files
    sci_fits_addresses = glob.glob(f'{data_dir}/{program_ID}/{object_ID}/*sci.fits') 
    
    # loop backwards through all filtered .sci files 
    for i in range(len(sci_fits_addresses) - 1, -1, -1): # loop through backwards
        filter = sci_fits_addresses[i].split('_')[-2]

        if filter in SW_filters:
            if SW_filters[filter][1] > shifted_LyAlpha:
                filter_list.append(filter)

    filter_list.reverse()
    return filter_list

 
def galight_prior_loop(object, data_dir, psf_dir, save_dir, npixels, nsigma):
    ''' Runs galight_prior on all images and their corresponding PSFs for the object.

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
        nsigma: float
            The value for Galight paramter nsigma used. The s/n defined to detect all the objects in the image stamp.
        npixels: int
            The value for Galight paramter nsigma used. The number of connected pixels, each greater than threshold, that an object must
            have to be detected.
    '''

    program_ID = object.split('_')[0]
    object_ID  = object.split('_')[1]

    # loop through PSFs first because there are less
    psf_filters = glob.glob(f'{psf_dir}/*0.04*') 
    img_filters = glob.glob(f'{data_dir}/{program_ID}/{object_ID}/*sci*') 
    for psf in psf_filters: 
        last_slash_index = psf.rfind('\\') # find the index of the last \  
        first_underscore_index = psf.find('_', last_slash_index) # find the index of the first _ after the last /
        filter = psf[last_slash_index + 1:first_underscore_index] # filter is between \ and _

        # check if there's an image with this filter 
        for img in img_filters:
            if filter + '_' in img and 'm' not in filter: # avoid convolved and m filters
                print(filter + ':')
                galight_prior(object, data_dir, psf_dir, save_dir, filter, nsigma, npixels)


def fit_object(object, data_dir, psf_dir, save_dir, nsigma, npixels, fit_all_filters=False):
    ''' Executes all necessary functions to fit all filters of an object. Assumes PSFs already exist.

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
        nsigma: float
            The value for Galight paramter nsigma used. The s/n defined to detect all the objects in the image stamp.
        npixels: int
            The value for Galight paramter nsigma used. The number of connected pixels, each greater than threshold, that an object must
            have to be detected.
        fit_all_filters: bool
            If fits for all of the object's filters should be created using galight_prior.
    '''

    z = get_z(object, data_dir)
    filter_list = get_filter_list(object, data_dir, z)

    make_stacked_PSF(object, data_dir, psf_dir, save_dir, filter_list)
    make_stacked_image(object, data_dir, save_dir, filter_list)

    galight_free(object, 'SW', save_dir, nsigma, npixels)
    
    if fit_all_filters:
        galight_prior_loop(object, data_dir, psf_dir, save_dir, nsigma, npixels)

 
def fit_all_objects(data_dir, psf_dir, save_dir, nsigma=1.9, npixels=4):
    ''' Calls fit_object on all objects in all programs.

    Parameters: 
    -----------
        data_dir: str 
            File path to where the program sci and wht .fits data is stored.
        psf_dir: str
            File path to where the .fits PSF files for filters are stored.
        save_dir: str
            File path to where the outputed stacked .fits files, .pdf plots, and .pkl dictionaries are saved to for all programs.
        nsigma: float
            The value for Galight paramter nsigma used. The s/n defined to detect all the objects in the image stamp.
        npixels: int
            The value for Galight paramter nsigma used. The number of connected pixels, each greater than threshold, that an object must
            have to be detected.
    '''

    programs = os.listdir(data_dir) 
    for program in sorted(programs, key=int): # sort in ascending order
        objects = os.listdir(f'{data_dir}/{program}')[0:-1] # exclude redshifts_dict.pkl
        for object in sorted(objects, key=int)[sorted(objects, key=int).index('0'):]: # sort in ascending order
            print(f'Fitting object {program}_{object}: ################################################################################')
            fit_object(f'{program}_{object}', data_dir, psf_dir, save_dir, nsigma, npixels)


def calc_all_weighted_residuals(data_dir, save_dir):
    ''' Calls the nessecary functions on all objects in all programs to calculate a weighted residual sum and save it to the objects 
    respective Galight dict. Assumes all objects have been initially fitted.
    
    Parameters: 
    -----------
        data_dir: str 
            File path to where the program sci and wht .fits data is stored.
        psf_dir: str
            File path to where the .fits PSF files for filters are stored.
        save_dir: str
            File path to where the outputed stacked .fits files, .pdf plots, and .pkl dictionaries are saved to for all programs.
    '''

    programs = os.listdir(data_dir) 
    for program in sorted(programs, key=int): # sort in ascending order
        objects = os.listdir(f'{data_dir}/{program}')[0:-1] # exclude redshifts_dict.pkl
        for object in sorted(objects, key=int)[sorted(objects, key=int).index('0'):]: # sort in ascending order            
            calc_weighted_residual(f'{program}_{object}', save_dir, results_dir)


def merge_plots(save_dir, program_ID, nsigma, npixels, output_dir):
    ''' Merge all SW final plots in a program as one PDF.

    Parameters: 
    -----------
        save_dir: str
            File path to where the outputed stacked .fits files, .pdf plots, and .pkl dictionaries are saved to for all programs.
        program_ID: str
            ID of the program to merge all plots for. 
        nsigma: float
            The value for Galight paramter nsigma used. The s/n defined to detect all the objects in the image stamp.
        npixels: int
            The value for Galight paramter nsigma used. The number of connected pixels, each greater than threshold, that an object must
            have to be detected.
        output_dir: str
            File path to where the merged plot is saved to.
    '''

    pdf_list = []

    objects = os.listdir(f'{save_dir}/{program_ID}')
    for object in sorted(objects, key=int):
        pdf_list.append(f'{save_dir}/{program_ID}/{object}/galight_SW_combined_plot.pdf')

    pdf_merger = PyPDF2.PdfMerger()

    for pdf in pdf_list:
        pdf_merger.append(pdf)

    with open(f'{output_dir}/{program_ID}_merged_pdf_{str(nsigma)}_{str(npixels)}.pdf', 'wb') as output_pdf:
        pdf_merger.write(output_pdf)


def merge_weighted_residuals(results_dir, program_ID, output_dir):
    ''' Save all weighted reisduals in a program to a .pkl dict.

    Parameters: 
    -----------
        results_dir: str
            File path to where the .pkl dictionaries containing the Sersic paramters for all objects are saved to.
        program_ID: str
            ID of the program to merge all plots for. 
        output_dir: str
            File path to where the merged plot is saved to.
    '''

    weighted_residuals_dict = {}
    dict_paths = glob.glob(f'{results_dir}/{program_ID}*_SW.pkl')

    # extract weighted residual from every object's results dict and store in new dict
    for dict_path in dict_paths:
        object = dict_path.split('_')[-2]

        with open(dict_path, 'rb') as handle:
            galight_dict = pkl.load(handle)
        
        weighted_residuals_dict[int(object)] = galight_dict['weighted residual']

    # sort dict in ascending order
    weighted_residuals_dict = dict(sorted(weighted_residuals_dict.items(), key=lambda item: int(item[0])))
            
    # save the new dict
    with open(f'{output_dir}/{program_ID}_weighted_residuals_dict.pkl', 'wb') as handle:
        pkl.dump(weighted_residuals_dict, handle)
    
    return 