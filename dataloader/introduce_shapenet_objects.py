import numpy as np
from scipy.spatial.transform import Rotation as R
from astropy import coordinates as ac
from astropy.coordinates import Angle


def load_shapenet_object(path):
    data = np.load(path)
    spn_points = data['points']
    return spn_points


def rotate_shapenet_object(spn_points, dx=0, dy=0, dz=0):
    r = R.from_euler('xyz', (dx, dy, dz), degrees=True)
    spn_points = r.apply(spn_points)
    return spn_points


def move_shapenet_object(spn_points, dx=0, dy=0, dz=0):
    spn_points[:, 0] += dx
    spn_points[:, 1] += dy
    spn_points[:, 2] += dz
    return spn_points

'''
description: 
param {*} spn_points
param {*} dlat
param {*} dlon
return {*}
'''
def move_shapenet_object_in_sphere(spn_points, dlat=0, dlon=0):
    # TODO: sanity check. make sure spn object does not pass through the origin
    """
    r: the radius from the origin point (0,0,0) is the input point.
    lat: 与x轴的夹角
    lon: 与z轴的夹角
    """

    r, lat, lon = ac.cartesian_to_spherical(spn_points[:, 0], spn_points[:, 1], spn_points[:, 2])
    lat += Angle(str(dlat) + 'd')
    lon += Angle(str(dlon) + 'd')
    x, y, z = ac.spherical_to_cartesian(r, lat, lon)

    return np.stack([x,y,z]).T


def resize_shapenet_object(spn_points, resize_factor=1):

    # find out object center
    object_center = np.mean(spn_points, axis=0)

    # calculate relative position w.r.t object center for each point
    distance_to_center = spn_points - object_center

    # resize the object
    spn_points = distance_to_center * resize_factor + object_center

    return spn_points

def put_shapenet_object_on_the_ground(spn_points, kitti_points):
    mean_x = spn_points[:, 0].mean()
    mean_y = spn_points[:, 1].mean()
    distance = np.square(kitti_points[:, 0] - mean_x) + np.square(kitti_points[:, 1] - mean_y)
    ground_height = kitti_points[:, 2][np.argmin(distance)]
    z_offset = spn_points[:, 2].min() - ground_height
    spn_points[:, 2] -= z_offset
    return spn_points


'''
description: 
param {*} spn_points: shapenet points, 
param {*} kitti_points:scene points
return {*}
'''
def add_shapenet_objects_to_kitti_points(spn_points, kitti_points):

    # Analyze kitti points
    r_map, lat_map, lon_map = ac.cartesian_to_spherical(kitti_points[:, 0], kitti_points[:, 1], kitti_points[:, 2])

    introduced_anomaly_indices = np.zeros(kitti_points.shape[0], dtype=bool)
    r, lat, lon = ac.cartesian_to_spherical(spn_points[:, 0], spn_points[:, 1], spn_points[:, 2])

    # no need to consider overflow in lat
    # need to consider overflow in lon

    # When overflow happens, x coordinates must be greater than 0
    x_overflow = spn_points[:, 0].min() > 0

    # When overflow happens, y's must have both negative and positive values
    lon_overflow = (spn_points[:, 1].min() < 0) and (spn_points[:, 1].max() > 0)

    # TODO: corner case, object on top of the camera (object passes the origin)
    if (spn_points[:, 0].min() < 0) and (spn_points[:, 0].max() > 0) and lon_overflow:
        # print('Object on top of the camera, skip this')
        return kitti_points, introduced_anomaly_indices

    overflow = x_overflow and lon_overflow


    if overflow:
        min_lon_index = spn_points[:, 1].argmin()
        max_lon_index = spn_points[:, 1].argmax()

        _, _, left_most_lon = ac.cartesian_to_spherical(spn_points[min_lon_index, 0], spn_points[min_lon_index, 1],
                                                        spn_points[min_lon_index, 2])
        _, _, rightmost_lon = ac.cartesian_to_spherical(spn_points[max_lon_index, 0], spn_points[max_lon_index, 1],
                                                        spn_points[max_lon_index, 2])

        # right half: 0 to the right edge of the object
        overlap_indices_right_half = (lon_map < rightmost_lon)
        overlap_indices_left_half = (lon_map > left_most_lon)

        # determine latitude overlap
        overlap_indices_lat = (lat_map > lat.min()) & (lat_map < lat.max())
        #!============================================
        overlap_indices_r =r_map > r.max()
        #!============================================
        overlap_indices = (overlap_indices_right_half | overlap_indices_left_half) & overlap_indices_lat & overlap_indices_r


    else:
        left_most_lon = lon.min()
        rightmost_lon = lon.max()

        overlap_indices_lat = (lat_map > lat.min()) & (lat_map < lat.max())
        overlap_indices_lon = (lon_map > left_most_lon) & (lon_map < rightmost_lon)

        #!============================================
        overlap_indices_r =r_map > r.max()
        #!============================================
        overlap_indices = overlap_indices_lon & overlap_indices_lat & overlap_indices_r


    # Update r's of kitti points occluded by the
    # newly introduced shapenet object
    #!+=====================================
    for i in np.where(overlap_indices)[0]:
    #!+=====================================

        """"
        # do nothing if current map point does not overlap with
        # the rectangle that encloses the shapenet object
        
        """
        

        # current map point's r_map, lat_map, lon_map
        curr_r_map, curr_lat_map, curr_lon_map = r_map[i], lat_map[i], lon_map[i]

        # find shapenet points that are close to the current map point
        th_lat = 0.2
        th_lon = 0.02
        mask_lat = np.abs(lat.degree - curr_lat_map.degree) < th_lat
        mask_lon = np.abs(lon.degree - curr_lon_map.degree) < th_lon
        mask = mask_lat & mask_lon

        # no nearby shapenet point, do not change r of current map point
        if mask.sum() == 0:
            continue

        # from nearby shapenet points, find out
        # which one has the shortest r
        index_of_shortest_r = r[mask].argmin()

        # replace r of current map point with the shortest r
        # of nearby shapenet point
        r_map[i] = r[mask][index_of_shortest_r]

        # mark current point as an introduced anomaly
        introduced_anomaly_indices[i] = True

    # update map points
    kitti_points[:, 0], kitti_points[:, 1], kitti_points[:, 2] = ac.spherical_to_cartesian(r_map, lat_map, lon_map)

    return kitti_points, introduced_anomaly_indices