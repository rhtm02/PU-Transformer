import numpy as np
import h5py
import torch

def permute_point_cloud(input_data):
    '''
        Permutation Points
        Input : [N,3]
        Output : permute - [N,3]
    '''
    N,_ = input_data.shape

    point_indice = np.arange(0,N)
    np.random.shuffle(point_indice)

    return input_data[point_indice,:]


def nonuniform_sampling(num, sample_num):
    sample = set()
    loc = np.random.rand() * 0.8 + 0.1
    while len(sample) < sample_num:
        a = int(np.random.normal(loc=loc, scale=0.3) * num)
        if a < 0 or a >= num:
            continue
        sample.add(a)
    return list(sample)


def rotate_point_cloud_and_gt(input_data, gt_data=None):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    angles = np.random.uniform(size=(3)) * 2 * np.pi
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

    input_data[:, :3] = np.dot(input_data[:, :3], rotation_matrix)
    if input_data.shape[1] > 3:
        input_data[:, 3:] = np.dot(input_data[:, 3:], rotation_matrix)

    if gt_data is not None:
        gt_data[:, :3] = np.dot(gt_data[:, :3], rotation_matrix)
        if gt_data.shape[1] > 3:
            gt_data[:, 3:] = np.dot(gt_data[:, 3:], rotation_matrix)

    return input_data, gt_data


def random_scale_point_cloud_and_gt(input_data, gt_data=None, scale_low=0.5, scale_high=2):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            Nx3 array, original point cloud
        Return:
            Nx3 array, scaled point cloud
    """
    scale = np.random.uniform(scale_low, scale_high)
    input_data[:, :3] *= scale
    if gt_data is not None:
        gt_data[:, :3] *= scale

    return input_data, gt_data, scale


def shift_point_cloud_and_gt(input_data, gt_data=None, shift_range=0.3):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, shifted point cloud
    """
    shifts = np.random.uniform(-shift_range, shift_range, 3)
    input_data[:, :3] += shifts
    if gt_data is not None:
        gt_data[:, :3] += shifts
    return input_data, gt_data


def jitter_perturbation_point_cloud(input_data, sigma=0.005, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, jittered point cloud
    """
    assert (clip > 0)
    jitter = np.clip(sigma * np.random.randn(*input_data.shape), -1 * clip, clip)
    jitter[:, 3:] = 0
    input_data += jitter
    return input_data


def rotate_perturbation_point_cloud(input_data, angle_sigma=0.03, angle_clip=0.09):
    """ Randomly perturb the point clouds by small rotations
        Input:
          Nx3 array, original point cloud
        Return:
          Nx3 array, rotated point cloud
    """
    angles = np.clip(angle_sigma * np.random.randn(3), -angle_clip, angle_clip)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    Ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    Rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))

    input_data[:, :3] = np.dot(input_data[:, :3], R)
    if input_data.shape[1] > 3:
        input_data[:, 3:] = np.dot(input_data[:, 3:], R)
    return input_data

def normalize_point_cloud(input):
    """
    input: pc [N, P, 3]
    output: pc, centroid, furthest_distance
    """
    if len(input.shape) == 2:
        axis = 0
    elif len(input.shape) == 3:
        axis = 1
    centroid = np.mean(input, axis=axis, keepdims=True)
    input = input - centroid
    furthest_distance = np.amax(
        np.sqrt(np.sum(input ** 2, axis=-1, keepdims=True)), axis=axis, keepdims=True)
    input = input / furthest_distance
    return input, centroid, furthest_distance



def load_h5_data(h5_filename='', opts=None, skip_rate=1, use_randominput=True):
    print("========== Loading Data ==========")
    num_point = opts.sampling_num_points
    num_4X_point = int(opts.sampling_num_points * 4)
    num_out_point = int(opts.sampling_num_points * opts.up_ratio)

    print("loading data from: {}".format(h5_filename))
    if use_randominput:
        print("use random input")
        with h5py.File(h5_filename, 'r') as f:
            input = f['poisson_%d' % num_4X_point][:]
            gt = f['poisson_%d' % num_out_point][:]
    else:
        print("Do not use random input")
        with h5py.File(h5_filename, 'r') as f:
            input = f['poisson_%d' % num_point][:]
            gt = f['poisson_%d' % num_out_point][:]

    # name = f['name'][:]
    assert len(input) == len(gt)

    print("Normalize the data")
    centroid = np.mean(input[:, :, 0:3], axis=1, keepdims=True)
    input[:, :, 0:3] = input[:, :, 0:3] - centroid
    furthest_distance = np.amax(np.sqrt(np.sum(input[:, :, 0:3] ** 2, axis=-1)), axis=1, keepdims=True)
    input[:, :, 0:3] = input[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    gt[:, :, 0:3] = gt[:, :, 0:3] - centroid
    gt[:, :, 0:3] = gt[:, :, 0:3] / np.expand_dims(furthest_distance, axis=-1)

    input = input[::skip_rate]
    gt = gt[::skip_rate]
    data_radius = furthest_distance[::skip_rate]
    print("total %d samples" % (len(input)))

    print("========== Finish Data Loading ========== \n")
    return input, gt, data_radius

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', default='../dataset/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5')
    parser.add_argument('--no_augment', action="store_false", dest="augment", default=True)
    parser.add_argument('--test_jitter', action='store_true', help='test with noise')
    parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
    parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")
    parser.add_argument('--num_point', type=int, default=256)
    parser.add_argument('--up_ratio', type=int, default=4)
    parser.add_argument('--patch_num_point', type=int, default=256)
    parser.add_argument('--patch_num_ratio', type=int, default=3)
    parser.add_argument('--fps', dest='random', action='store_false', default=True,
                        help='use random input, or farthest sample input(default)')

    args = parser.parse_args()
    input,gt,radius = load_h5_data(args.data_dir, opts=args, skip_rate=1)
    print(input.shape,gt.shape,radius.shape)
    print((input - gt).sum())