import torch
import numpy as np
import data.data_utils as utils


class CUDAPrefetcher:
    """Use the CUDA side to accelerate data reading.
    Args:
        dataloader (DataLoader): Data loader. Combines a dataset and a sampler, and provides an iterable over the given dataset.
        device (torch.device): Specify running device.
    """

    def __init__(self, dataloader, device: torch.device):
        self.batch_data = None
        self.original_dataloader = dataloader
        self.device = device

        self.data = iter(dataloader)
        self.stream = torch.cuda.Stream()
        self.preload()

    def preload(self):
        try:
            self.batch_data = next(self.data)
        except StopIteration:
            self.batch_data = None
            return None

        with torch.cuda.stream(self.stream):
            for k, v in self.batch_data.items():
                if torch.is_tensor(v):
                    self.batch_data[k] = self.batch_data[k].to(self.device, non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        batch_data = self.batch_data
        self.preload()
        return batch_data

    def reset(self):
        self.data = iter(self.original_dataloader)
        self.preload()

    def __len__(self) -> int:
        return len(self.original_dataloader)

class DataLoader(torch.utils.data.Dataset):
    def __init__(self, opts,skip_rate=1):
        super().__init__()
        self.opts = opts

        self.input_data, self.gt_data, self.radius_data = utils.load_h5_data(self.opts.data_path, opts=self.opts,
                                                                       use_randominput=self.opts.use_random_input,
                                                                             skip_rate=skip_rate)

    def __len__(self):
        return self.input_data.shape[0]

    def __getitem__(self, index):
        input_data = self.input_data[index]
        gt_data = self.gt_data[index]
        radius_data = np.array([self.radius_data[index]])

        sample_idx = utils.nonuniform_sampling(self.input_data.shape[1],sample_num=self.opts.sampling_num_points)
        input_data = input_data[sample_idx, :]

        # for data aug
        input_data, gt_data = utils.rotate_point_cloud_and_gt(input_data, gt_data)
        input_data, gt_data, scale = utils.random_scale_point_cloud_and_gt(input_data, gt_data,
                                                                           scale_low=0.9, scale_high=1.1)
        input_data, gt_data = utils.shift_point_cloud_and_gt(input_data, gt_data, shift_range=0.1)
        radius_data = radius_data * scale
        # print(input_data.shape)

        if np.random.rand() > 0.7:
            input_data = utils.jitter_perturbation_point_cloud(input_data, sigma=self.opts.jitter_sigma, clip=self.opts.jitter_max)
        if np.random.rand() > 0.7:
            input_data = utils.rotate_perturbation_point_cloud(input_data, angle_sigma=self.opts.angle_sigma, angle_clip=self.opts.angle_max)
        if np.random.rand() > 0.7:
            input_data = utils.permute_point_cloud(input_data)
        return input_data, gt_data, radius_data

if __name__=="__main__":
    import argparse
    import time
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',
                        default='../dataset/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5')

    parser.add_argument('--use_random_input',  type=bool, default=True, help="permute augmentation")

    parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
    parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")

    parser.add_argument('--angle_sigma', type=float, default=0.01, help="angle augmentation")
    parser.add_argument('--angle_max', type=float, default=0.03, help="angle augmentation")

    parser.add_argument('--permute_point', type=bool, default=True, help="permute augmentation")

    parser.add_argument('--sampling_num_points', type=int, default=256)
    parser.add_argument('--up_ratio', type=int, default=4)

    parser.add_argument('--fps', dest='random', action='store_false', default=True,
                        help='use random input, or farthest sample input(default)')

    args = parser.parse_args()

    num_workers = 10
    s = time.time()
    loader = DataLoader(args)
    dataset = torch.utils.data.DataLoader(dataset=loader, batch_size=32, shuffle=True,
                    num_workers=num_workers, pin_memory=True, drop_last=True)

    for idx,(inputs,gt,radius) in enumerate(dataset):

        print(idx,inputs.shape,gt.shape,radius.shape)
        if idx > 10:
            break
    print(time.time() - s)
    #(input_data,gt_data,radius_data)=dataset.__getitem__(0)
    #print(input_data.shape,gt_data.shape,radius_data.shape)
    #dataset=PUNET_Dataset_Whole(data_dir="../MC_5k",n_input=1024)
    #points=dataset.__getitem__(0)
    #print(points.shape)