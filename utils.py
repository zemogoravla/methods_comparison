import numpy as np
import rasterio
import rpcm
import utm
import s2p




def robust_image_min_max(I, lower_percentile=2, upper_percentile=98):
    min = np.percentile(I, lower_percentile)
    max = np.percentile(I, upper_percentile)
    return min, max


def readGTIFFmeta(fname):
    """
    Reads the image GeoTIFF metadata using rasterio and returns it,
    along with the bounding box, in a tuple: (meta, bounds)
    if the file format doesn't support metadata the returned metadata is invalid
    This is the metadata rasterio was capable to interpret,
    but the ultimate command for reading metadata is *gdalinfo*
    """
    with  rasterio.open(fname, 'r') as s:
        ## interesting information
        # print(s.crs,s.meta,s.bounds)
        return (s.meta,s.bounds)

def get_angles(ref_filename, sec_filename):
    ref_rpc = rpcm.rpc_from_geotiff(ref_filename)
    sec_rpc = rpcm.rpc_from_geotiff(sec_filename)

    width, height, pixel_dim = s2p.common.image_size_gdal(ref_filename)

    # get the rpc of the first image
    ref_image_rpc = rpcm.rpc_from_geotiff(ref_filename)
    # get the localization of the center of the image
    lon_center, lat_center = ref_image_rpc.localization(width // 2, height // 2, 0)

    ref_zenith, ref_azimut = ref_rpc.incidence_angles(lon_center, lat_center, z=0)
    sec_zenith, sec_azimut = sec_rpc.incidence_angles(lon_center, lat_center, z=0)

    ref_sec_angle = rpcm.angle_between_views(ref_filename, sec_filename, lon_center, lat_center, z=0)

    return ref_zenith, ref_azimut, sec_zenith, sec_azimut, ref_sec_angle


def robust_image_min_max(I, lower_percentile=0.001, upper_percentile=99.999):
    min = np.percentile(I, lower_percentile)
    max = np.percentile(I, upper_percentile)
    return min, max

def get_min_max_height_from_gt(gt_filename):
    # get min and max height from the gt image
    gt = s2p.common.gdal_read_as_array_with_nans(gt_filename)
    min_height, max_height = robust_image_min_max(gt)
    return min_height, max_height


def aoi_info_from_geotiff_gt(ref_filename, gt_filename, padding=0.0, height_guard=[-5,+5]):
    ''' aoi_from_gt
    Returns an aoi from an image and a gt

    padding: relative to the size the gt region (e.g. 0.1 adds a 10% of the extension of
    the region on the four boundaries)
    '''

    # get the dimensions of the first image
    width, height, pixel_dim = s2p.common.image_size_gdal(ref_filename)

    # get the rpc of the first image
    ref_image_rpc = rpcm.rpc_from_geotiff(ref_filename)
    # get the localization of the center of the image
    lon_center, lat_center = ref_image_rpc.localization(width // 2, height // 2, 0)

    zone_number = utm.latlon_to_zone_number(lat_center, lon_center)
    zone_letter = utm.latitude_to_zone_letter(lat_center)


    # Build the AOI of the GT ------------------------------

    gt_metadata = readGTIFFmeta(gt_filename)
    bounding_box = gt_metadata[1]
    gt_min_easting = bounding_box.left
    gt_max_easting = bounding_box.right
    gt_min_northing = bounding_box.bottom
    gt_max_northing = bounding_box.top

    # padding
    gt_easting_extension = gt_max_easting - gt_min_easting
    gt_northing_extension = gt_max_northing - gt_min_northing
    gt_min_easting -= padding * gt_easting_extension
    gt_max_easting += padding * gt_easting_extension
    gt_min_northing -= padding * gt_northing_extension
    gt_max_northing += padding * gt_northing_extension
    # update the extension
    gt_easting_extension = gt_max_easting - gt_min_easting
    gt_northing_extension = gt_max_northing - gt_min_northing

    # convert easting, northing to lon,lat
    gt_min_lat, gt_min_lon = utm.to_latlon(gt_min_easting, gt_min_northing, zone_number, zone_letter)
    gt_max_lat, gt_max_lon = utm.to_latlon(gt_max_easting, gt_max_northing, zone_number, zone_letter)

    zone_hemisphere = 'N' if gt_min_lat > 0 else 'S'

    aoi = {'coordinates': [[[gt_min_lon, gt_min_lat],
                           [gt_min_lon, gt_max_lat],
                           [gt_max_lon, gt_max_lat],
                           [gt_max_lon, gt_min_lat],
                           [gt_min_lon, gt_min_lat]]], 'type': 'Polygon'}

    utm_bbx = [gt_min_easting, gt_max_easting, gt_min_northing, gt_max_northing]
    lonlat_bbx = [gt_min_lon, gt_max_lon, gt_min_lat, gt_max_lat]

    # get min and max height from the gt image
    gt = s2p.common.gdal_read_as_array_with_nans(gt_filename)
    min_height, max_height = robust_image_min_max(gt)
    # add height guard
    min_height += height_guard[0]
    max_height += height_guard[1]

    return aoi, min_height, max_height, zone_hemisphere, zone_letter, zone_number, utm_bbx, lonlat_bbx



# image normalization as done by VisSatSatelliteStereo
def normalize_image(I, gamma=1/2.2, lower_percentile=2 , upper_percentile=98):
    J = I.copy().astype(np.double)
    J = J**gamma
    lower = np.percentile(J, lower_percentile)
    upper = np.percentile(J, upper_percentile)
    J = (J - lower) / (upper - lower)
    J[J < 0] = 0
    J[J > 1] = 1
    return (J * 255).astype('uint8')

# image normalization as done by VisSatSatelliteStereo (considering nans)
def normalize_image_accepts_nans(I, gamma=1/2.2, lower_percentile=2 , upper_percentile=98):
    J = I.copy().astype(np.double)
    J = J**gamma
    lower = np.nanpercentile(J, lower_percentile)
    upper = np.nanpercentile(J, upper_percentile)
    J = (J - lower) / (upper - lower)
    J[J < 0] = 0
    J[J > 1] = 1
    return (J * 255).astype('uint8')


# translate names between s2p and VisSatSatelliteStereo
# See: VisSatSatelliteStereo/lib/parse_meta.py and VisSatSatelliteStereo/lib/rpc_model.py

def vissat_meta_from_geotiff(image_filename):
    geotiff_meta = readGTIFFmeta(image_filename)
    rpc = rpcm.rpc_from_geotiff(image_filename)

    meta_dict = {}
    rpc_dict = {}

    rpc_dict['rowOff'] = rpc.row_offset
    rpc_dict['rowScale'] = rpc.row_scale

    rpc_dict['colOff'] = rpc.col_offset
    rpc_dict['colScale'] = rpc.col_scale

    rpc_dict['latOff'] = rpc.lat_offset
    rpc_dict['latScale'] = rpc.lat_scale

    rpc_dict['lonOff'] = rpc.lon_offset
    rpc_dict['lonScale'] = rpc.lon_scale

    rpc_dict['altOff'] = rpc.alt_offset
    rpc_dict['altScale'] = rpc.alt_scale

    # polynomial coefficients
    rpc_dict['rowNum'] = rpc.row_num
    rpc_dict['rowDen'] = rpc.row_den
    rpc_dict['colNum'] = rpc.col_num
    rpc_dict['colDen'] = rpc.col_den

    # width, height
    meta_dict['width'] = geotiff_meta[0]['width']
    meta_dict['height'] = geotiff_meta[0]['height']

    meta_dict['rpc'] = rpc_dict

    return meta_dict


def get_all_possible_pairs_from_list(L, order_matters=False):
    # https://stackoverflow.com/questions/18201690/get-unique-combinations-of-elements-from-a-python-list
    from itertools import combinations, permutations
    if order_matters:
        all_pairs = [list(comb) for comb in permutations(L,2)]
    else:
        all_pairs = [list(comb) for comb in combinations(L,2)]
    return all_pairs


if __name__ == '__main__':
    from skimage.io import imread, imsave
    import argparse

    parser = argparse.ArgumentParser(description='Open, normalize to 8bit range and save')
    parser.add_argument('input_img_filename', type=str)
    parser.add_argument('output_img_filename', type=str)
    parser.add_argument('--top', type=int, default=0)
    parser.add_argument('--left', type=int, default=0)
    parser.add_argument('--width', type=int, default=-1)
    parser.add_argument('--height', type=int, default=-1)

    opt = parser.parse_args()

    I = imread(opt.input_img_filename)
    J = normalize_image_accepts_nans(I)

    if opt.width == -1:
        opt.width = I.shape[1]
    if opt.height == -1:
        opt.height = I.shape[0]

    J = J[opt.top:opt.top+opt.height, opt.left:opt.left+opt.width]
    imsave(opt.output_img_filename, J)
