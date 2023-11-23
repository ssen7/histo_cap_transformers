from openslide import open_slide
import histolab

from histolab.tiler import RandomTiler, GridTiler
from histolab.slide import Slide

from histolab.filters.morphological_filters import BinaryDilation

from typing import Any, Callable, List, Tuple
from skimage.measure import label, regionprops
import numpy as np
import h5py

from collections import namedtuple

CoordinatePair = namedtuple("CoordinatePair", ("x_ul", "y_ul", "x_br", "y_br"))
Region = namedtuple("Region", ("index", "area", "bbox", "center", "coords"))

def regions_from_binary_mask(binary_mask):
    """Calculate regions properties from a binary mask.
    Parameters
    ----------
    binary_mask : np.ndarray
        Binary mask from which to extract the regions
    Returns
    -------
    List[Region]
        Properties for all the regions present in the binary mask
    """

    def convert_np_coords_to_pil_coords(
        bbox_np: Tuple[int, int, int, int]
    ) -> Tuple[int, int, int, int]:
        return (*reversed(bbox_np[:2]), *reversed(bbox_np[2:]))

    thumb_labeled_regions = label(binary_mask)
    regions = [
        Region(
            index=i,
            area=rp.area,
            bbox=convert_np_coords_to_pil_coords(rp.bbox),
            center=rp.centroid,
            coords=rp.coords,
        )
        for i, rp in enumerate(regionprops(thumb_labeled_regions))
    ]
    return regions

def scale_coordinates(
    reference_coords: CoordinatePair,
    reference_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> CoordinatePair:
    """Compute the coordinates corresponding to a scaled version of the image.
    Parameters
    ----------
    reference_coords: CoordinatePair
        Coordinates referring to the upper left and lower right corners
        respectively.
    reference_size: tuple of int
        Reference (width, height) size to which input coordinates refer to
    target_size: tuple of int
        Target (width, height) size of the resulting scaled image
    Returns
    -------
    coords: CoordinatesPair
        Coordinates in the scaled image
    """
    reference_coords = np.asarray(reference_coords).ravel()
    reference_size = np.tile(reference_size, 2)
    target_size = np.tile(target_size, 2)
    return CoordinatePair(
        *np.floor((reference_coords * target_size) / reference_size).astype("int64")
    )

def create_patches_fp(slide_path, save_path, size=4096):

    slide_name = slide_path.split('/')[-1].split('.')[0]

    slide = Slide(slide_path,processed_path=save_path)
    all_tissue_mask = histolab.masks.TissueMask()

    mask=all_tissue_mask(slide)

    number_of_tissues = len(regions_from_binary_mask(mask))
    coords_list = []
    for i in range(number_of_tissues):
        levelbbox=CoordinatePair(*regions_from_binary_mask(mask)[i].bbox)
        bbox_coordinates_lvl = scale_coordinates(levelbbox, mask.shape[::-1], slide.level_dimensions(0))
        coords_list += get_coords(bbox_coordinates_lvl, size)

    with h5py.File(f'{save_path}/{slide_name}.h5', 'w') as f:
        dset = f.create_dataset('coords', data=coords_list)


def get_coords(bbox_coordinates_lvl, size=4096):
    x_ul = bbox_coordinates_lvl.x_ul
    y_ul = bbox_coordinates_lvl.y_ul
    x_br = bbox_coordinates_lvl.x_br
    y_br = bbox_coordinates_lvl.y_br

    coords = []
    for x in np.arange(x_ul, x_br, size):
        for y in np.arange(y_ul, y_br, size):
            coords += [(x, y)]

    return coords
    


    

