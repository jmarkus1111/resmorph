''' Functions that create variance weighted stacked PSF and sci images from a list of filters for a given object.
'''

import glob
import astropy.io.fits as fits
import os
from astropy.nddata import Cutout2D
from astropy import wcs

def make_stacked_image(object, data_dir, save_dir, filter_list):
    ''' Creates a stacked image of an object by doing a variance weighted sum of a list of .fits PSF files for different filters.

    Parameters: 
    -----------
        object: str 
            The program ID and object ID of the object being fitted in the form f'{program_ID}_{object_ID}'.
        data_dir: str
            File path to where NIRCam and MIRI data are stored for all programs.
        save_dir: str
            File path to where the outputed stacked .fits files, .pdf plots, and .pkl dictionaries are saved to for all programs.
        filter_list: list of str
            Filters who's respective .fits sci and wht files are used to create the stacked image.
    
    Returns:
    --------
        sum_sci_fits_file: str
            File path of the stacked image as a .fits file.
        sum_wht_fits_file: str
            File path of the stacked wht as a .fits file.
    '''

    # config
    program_ID = object.split('_')[0]
    object_ID  = object.split('_')[1]

    position = (500, 500) # center of 1000x1000 image
    size = (80, 80)

    # get list of files
    sci_fits_addresses = glob.glob(f'{data_dir}/{program_ID}/{object_ID}/*sci.fits')
    wht_fits_addresses = glob.glob(f'{data_dir}/{program_ID}/{object_ID}/*wht.fits')
    
    # select only JWST SW unconvolved filters
    for i in range(len(sci_fits_addresses) - 1, -1, -1): # loop through backwards through all filtered .sci files 
        filter = sci_fits_addresses[i].split('_')[-2]
        if filter not in filter_list:
            del sci_fits_addresses[i]

    # sum files
    sum_sci_data = 0
    sum_wht_data = 0
    for i in range(len(sci_fits_addresses)):
        # sci cutout
        sci_fits_file = fits.open(sci_fits_addresses[i])
        sci_cutout = Cutout2D(sci_fits_file[0].data, position, size, wcs=wcs.WCS(sci_fits_file[0].header))
        sci_cutout_data = sci_cutout.data
        sci_cutout_header = sci_cutout.wcs.to_header()

        # wht cutout
        wht_fits_file = fits.open(wht_fits_addresses[i])
        wht_cutout = Cutout2D(wht_fits_file[0].data, position, size, wcs=wcs.WCS(wht_fits_file[0].header))
        wht_cutout_data = wht_cutout.data
        wht_cutout_header = wht_cutout.wcs.to_header()

        # sum data
        sum_sci_data += sci_cutout_data*wht_cutout_data
        sum_wht_data += wht_cutout_data

        # reading the header for the weighted variance image
        sum_sci_header = sci_cutout_header
        sum_wht_header = wht_cutout_header
        missed_keys = ['PHOTFLAM', 'PHOTPLAM']
        if 'f090w' in sci_fits_addresses[i]: # not sure the intention of this 
            for key in missed_keys:
                sum_sci_header[key] = sci_fits_file[0].header[key]

    # normalize
    sum_sci_data = sum_sci_data/sum_wht_data

    # make sum fits files
    sum_sci_fits_file = fits.PrimaryHDU(sum_sci_data, header=sum_sci_header)
    sum_wht_fits_file = fits.PrimaryHDU(sum_wht_data, header=sum_wht_header)

    # save
    if not os.path.exists(f'{save_dir}/{program_ID}/{object_ID}'): os.mkdir(f'{save_dir}/{program_ID}/{object_ID}')

    type = 'sci'
    cutout_path = f'{save_dir}/{program_ID}/{object_ID}/{object}_{type}.fits'
    sum_sci_fits_file.writeto(cutout_path, overwrite=True)

    type = 'wht'
    cutout_path = f'{save_dir}/{program_ID}/{object_ID}/{object}_{type}.fits'
    sum_wht_fits_file.writeto(cutout_path, overwrite=True)

    return sum_sci_fits_file, sum_wht_fits_file


def make_stacked_PSF(object, data_dir, psf_dir, save_dir, filter_list):
    ''' Creates a stacked PSF for an object by doing a variance weighted sum of a list of .fits PSF files for different filters.

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
        filter_list: list of str
            Filters who's respective .fits PSF and wht files are used to create the stacked PSF.
    
    Returns:
    --------
        sum_psf_file: str
            File path of the stacked PSF as a .fits file.
    '''

    # config
    program_ID = object.split('_')[0]
    object_ID  = object.split('_')[1]

    position = (500, 500) # center of 1000x1000 image
    size = (99, 99)

    # get list of psfs and corresponding whts 
    psfs = []
    whts = []
    for filter in filter_list:
        psfs.append(glob.glob(f'{psf_dir}/{filter}*0.04*')[0])
        whts.append(glob.glob(f'{data_dir}/{program_ID}/{object_ID}/*{filter}_wht*')[0])

    # sum psfs weighted by whts 
    sum_psf_data = 0
    sum_wht_data = 0
    for i in range(len(psfs)):
        psf_file = fits.open(psfs[i])
        wht_file = fits.open(whts[i])

        psf_data = psf_file[0].data
        wht_data = Cutout2D(wht_file[0].data, position, size, wcs=wcs.WCS(wht_file[0].header)).data

        sum_psf_data += psf_data*wht_data
        sum_wht_data += wht_data

    # normalize
    sum_psf_data = sum_psf_data / sum_wht_data

    # make sum fits files
    sum_psf_file = fits.PrimaryHDU(sum_psf_data, header=psf_file[0].header)

    # save
    if not os.path.exists(f'{save_dir}/{program_ID}/{object_ID}'): os.mkdir(f'{save_dir}/{program_ID}/{object_ID}')
    
    cutout_path = f'{save_dir}/{program_ID}/{object_ID}/SW_0.04.fits'
    sum_psf_file.writeto(cutout_path, overwrite=True)

    return sum_psf_file