''' Functions related to calculating the "weighted residual".

    calc_weighted residual should be exectued with an object that has been previously fitted.
'''

import pickle as pkl
import numpy as np

def get_fitting_arrays(object, save_dir, results_dir):
    ''' Gets the segmentation map, data image, and model image for a fitted object.

    Parameters: 
    -----------
        object: str 
            The program ID and object ID of the object to get data from in the form f'{program_ID}_{object_ID}'.
        save_dir: str
            File path to where the outputed stacked .fits files, .pdf plots, and .pkl dictionaries are saved to for all programs.
        results_dir: str
            File path to where the .pkl dictionaries containing the Sersic paramters for all objects are saved to.

    Returns:
    --------
        segmentation_map: array of arrays of ints
            2d array of integers representing which object each pixel belongs to as determined by Galight. 0 corresponds to no object, 
            1 to the central object, and higher numbers to additional objects in the image.
        data: array of arrays of floats 
            2d array of the original photometry data.
        model: array of arrays of floats 
            2d array of the reconstructed image based on the Galight fit.
        central_model: array of arrays of floats 
            2d array of the reconstructed image based on the Galight fit of the central object only.
    '''

    program_ID = object.split('_')[0]
    object_ID  = object.split('_')[1]

    # get segmentation map array from detection dict in saves
    with open(f'{save_dir}/{program_ID}/{object_ID}/detection_dict.pkl', 'rb') as handle:
        detection_dict = pkl.load(handle)

    segmentation_map = detection_dict['segm_deblend']

    # get data and model arrays from object_SW dict in results
    with open(f'{results_dir}/{program_ID}_{object_ID}_SW.pkl', 'rb') as handle:
            galight_dict = pkl.load(handle)

    data = galight_dict['data']
    model = galight_dict['model']
    central_model = galight_dict['central model']
    noise = galight_dict['noise']

    return segmentation_map, data, model, central_model, noise


def find_adjacent_objects(segmentation_map):
    ''' Finds the ids of objects in the cutout bordering the central object. 

    Parameters: 
    -----------
        segmentation_map: array of arrays of ints
            2d array of integers representing which object each pixel belongs to as determined by Galight. 0 corresponds to no object, 
            1 to the central object, and higher numbers to additional objects in the image.

    Returns:
    --------
        adjacent_objects: list of ints
            List of object ids in the cutout which have at least on pixel adjacent to a pixel of the central object.
    '''

    num_rows = len(segmentation_map)
    num_cols = len(segmentation_map[0])
    adjacent_objects = set()

    # check left, right, above, below 
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    for row in range(num_rows):
        for col in range(num_cols):
            if segmentation_map[row][col] == 1:
                for hor_shift, vert_shift in directions:
                    new_row, new_col = row + hor_shift, col + vert_shift # find coords of adjacent element 

                    # check if coords of the element are within the array and add it to set if not 0 or 1
                    if 0 <= new_row < num_rows and 0 <= new_col < num_cols: 
                        if segmentation_map[new_row][new_col] > 1:
                            adjacent_objects.add(segmentation_map[new_row][new_col] - 1) # correct for different indexing in array

    return list(adjacent_objects)


def find_all_objects(segmentation_map):
    ''' Finds the ids of all objects in the cutout other than the central object. 

    Parameters: 
    -----------
        segmentation_map: array of arrays of ints
            2d array of integers representing which object each pixel belongs to as determined by Galight. 0 corresponds to no object, 
            1 to the central object, and higher numbers to additional objects in the image.

    Returns:
    --------
        all_objects: list of ints
            List of object ids of of all objects in the cutout other than the central object.
    '''

    num_objects = max(max(row) for row in segmentation_map)
    all_objects = list(range(1, num_objects)) # doesn't include central object 0

    return all_objects


def calc_weighted_residual(object, save_dir, results_dir):
    ''' Calculates the "weighted residual" for a given object. Assumes the object has already been fited. The weighted 
    residual is a single number that quantifies how well the central and nearby objects are described by the Sersic profile.

    Parameters: 
    -----------
        object: str 
            The program ID and object ID of the object to calculate its weighted residual in the form f'{program_ID}_{object_ID}'.
        save_dir: str
            File path to where the outputed stacked .fits files, .pdf plots, and .pkl dictionaries are saved to for all programs.
        results_dir: str
            File path to where the .pkl dictionaries containing the Sersic paramters for all objects are saved to.

    Returns:
    --------
        weighted_residual: float
            Sum of the absolute values of the residuals for all pixels contained within the central object and its adjacent objects. The 
            residual is calculated by subtracting the model of only the central object from the original data, and normalizing by the 
            same central model.
    '''
    
    program_ID = object.split('_')[0]
    object_ID  = object.split('_')[1]

    # get new model from only the central fit
    segmentation_map, data, model, central_model, noise = get_fitting_arrays(object, save_dir, results_dir)

    adjacent_objects = find_adjacent_objects(segmentation_map)

    # calculate weighted residual and sum
    weighted_residual = 0
    for i in range(len(segmentation_map)):
        for j in range(len(segmentation_map[0])):
            if segmentation_map[i][j] - 1 in adjacent_objects or segmentation_map[i][j] - 1 == 0:
                # sum = |(data - central model)| / central model
                weighted_residual += np.abs((data[i][j] - central_model[i][j])) / central_model[i][j]

    # save the summed residual in the object's SW dict

    with open(f'{results_dir}/{program_ID}_{object_ID}_SW.pkl', 'rb') as handle:
        galight_dict = pkl.load(handle)
    
    galight_dict['weighted residual'] = weighted_residual

    with open(f'{results_dir}/{program_ID}_{object_ID}_SW.pkl', 'wb') as handle:
        pkl.dump(galight_dict, handle)
                
    print(f'{program_ID}_{object_ID}: {weighted_residual}')
    return weighted_residual
