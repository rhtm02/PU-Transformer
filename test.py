import torch
import torch.nn as nn
import argparse

from losses.ChamferDistanceLoss import ChamferDistance
from losses.EMDLoss import earth_mover_distance
from losses.UniformLoss import get_uniform_loss
from models.PointTransformer import Upsampling
from data.dataloader import DataLoader
from models.PuTransformer import PointUpsamplingTransformer

parser = argparse.ArgumentParser()
parser.add_argument('--data_path',
                    default='./dataset/train/pu1k_poisson_256_poisson_1024_pc_2500_patch50_addpugan.h5')

parser.add_argument('--use_random_input',  type=bool, default=True, help="permute augmentation")

parser.add_argument('--jitter_sigma', type=float, default=0.01, help="jitter augmentation")
parser.add_argument('--jitter_max', type=float, default=0.03, help="jitter augmentation")

parser.add_argument('--angle_sigma', type=float, default=0.01, help="angle augmentation")
parser.add_argument('--angle_max', type=float, default=0.03, help="angle augmentation")

parser.add_argument('--permute_point', type=bool, default=True, help="permute augmentation")

parser.add_argument('--sampling_num_points', type=int, default=256)
parser.add_argument('--up_ratio', type=int, default=4)

parser.add_argument('--sampling_std', type=float, default=0.05)

parser.add_argument('--num_workers', type=int, default=10)

parser.add_argument('--batch_size', type=int, default=64)

parser.add_argument('--epochs', type=int, default=50)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0001)

parser.add_argument('--uniform_w', type=float, default=1.0)
parser.add_argument('--chamfer_w', type=float, default=1.0)
parser.add_argument('--emd_w', type=float, default=1.0)

parser.add_argument('--layer', type=int, default=6)
parser.add_argument('--feature', type=list, default=[32,128,512,1024,512,128])
parser.add_argument('--save_dir', type=str, default='./model')
parser.add_argument('--k', type=int, default=8)

args = parser.parse_args()

K = 8

loader = DataLoader(args,skip_rate=10)
dataset = torch.utils.data.DataLoader(dataset=loader, batch_size=args.batch_size, shuffle=True,
                                      num_workers=15, pin_memory=True, drop_last=True)

for PATH in [f'./k-{20}_putransformer_6900/']:

    model = PointUpsamplingTransformer(in_channels=3, out_channels=3, upscale_factor=4).cuda()

    model.load_state_dict(torch.load(PATH + 'best_model.pth'))
    print("Load Dic")


    if torch.cuda.is_available():
        model.cuda()

    cnt = 0
    emd_loss = 0
    uniform_loss = 0
    for idx,(sparse_points,dense_points,_) in  enumerate(dataset):
        cnt += 1

        if torch.cuda.is_available():
            sparse_points = sparse_points.cuda()
            dense_points = dense_points.cuda()

        with torch.no_grad():
            dense_output = model(sparse_points)

            emd_loss_ = earth_mover_distance(dense_output.permute(0, 2, 1), dense_points.permute(0, 2, 1))
            uniform_loss_ = get_uniform_loss(dense_output)

        emd_loss += torch.mean(emd_loss_)
        uniform_loss += uniform_loss_

    print(PATH,f'EMD : {emd_loss/cnt} Uniform : {uniform_loss/cnt}')


'''
batch - 80
./k-8_error/ EMD : 128.625244140625 Uniform : 0.012250741012394428
./k-8_error_none_positonal/ EMD : 99.56546020507812 Uniform : 0.01093081571161747
./k-8_regression/ EMD : 2656.416748046875 Uniform : 0.02446320652961731
./k-8_error_none_positonal_sphere/ EMD : 96.8829574584961 Uniform : 0.01293914020061493
'''
'''
{64:10,32:20,16:40,8:80,4:80,2:80}
./k-64_error_none_positonal/ EMD : 79.83177185058594 Uniform : 0.00996155384927988
Load Dic
./k-32_error_none_positonal/ EMD : 76.7293472290039 Uniform : 0.009532458148896694
Load Dic
./k-16_error_none_positonal/ EMD : 97.72833251953125 Uniform : 0.010517283342778683
Load Dic
./k-8_error_none_positonal/ EMD : 100.69551849365234 Uniform : 0.01122979260981083
Load Dic
./k-4_error_none_positonal/ EMD : 107.56661987304688 Uniform : 0.012045346200466156
Load Dic
./k-2_error_none_positonal/ EMD : 94.20866394042969 Uniform : 0.013531647622585297
'''

'''
batch - 20
Load Dic
./k-64_error_none_positonal/ EMD : 78.8099365234375 Uniform : 0.009828832931816578
Load Dic
./k-32_error_none_positonal/ EMD : 78.55596160888672 Uniform : 0.00945181492716074
Load Dic
./k-16_error_none_positonal/ EMD : 104.53926086425781 Uniform : 0.010537083260715008
Load Dic
./k-8_error_none_positonal/ EMD : 109.65966033935547 Uniform : 0.011191928759217262
Load Dic
./k-4_error_none_positonal/ EMD : 115.96453857421875 Uniform : 0.012203754857182503
Load Dic
./k-2_error_none_positonal/ EMD : 100.85398864746094 Uniform : 0.01357863750308752
'''
'''
Load Dic
./k-8_error_none_positonal_std-0.01/ EMD : 91.43058776855469 Uniform : 0.011977137066423893
Load Dic
./k-8_error_none_positonal_std-0.03/ EMD : 95.5618667602539 Uniform : 0.010984952561557293
Load Dic
./k-8_error_none_positonal_std-0.09/ EMD : 107.14136505126953 Uniform : 0.010848859325051308
Load Dic
./k-8_error_none_positonal_std-0.27/ EMD : 121.75202941894531 Uniform : 0.011674044653773308
Load Dic
./k-8_error_none_positonal_std-0.81/ EMD : 168.2052764892578 Uniform : 0.013349071145057678
Load Dic
./k-8_error_none_positonal_std-1/ EMD : 182.61788940429688 Uniform : 0.012840730138123035
'''

'''
pu transformer

'''