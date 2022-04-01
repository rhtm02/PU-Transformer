# Implemnet code
import torch
import torch.nn as nn
import argparse
import os
import torch.optim.lr_scheduler as lr_scheduler

from losses.ChamferDistanceLoss import ChamferDistance
from losses.EMDLoss import earth_mover_distance
from losses.UniformLoss import get_uniform_loss
from models.PointTransformer import Upsampling,UpsamplingPT,count_parameters
from models.PuTransformer import PointUpsamplingTransformer
from data.dataloader import DataLoader
from torch.cuda import amp

def xavier_init(m):
    classname = m.__class__.__name__
    #print(classname)
    if classname.find('Conv') != -1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('Linear')!=-1:
        nn.init.xavier_normal(m.weight)
    elif classname.find('BatchNorm') != -1:
        nn.init.constant_(m.weight, 1)
        nn.init.constant_(m.bias, 0)



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

parser.add_argument('--epochs', type=int, default=100)

parser.add_argument('--lr', type=float, default=0.0001)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--weight_decay', type=float, default=0.0001)

parser.add_argument('--uniform_w', type=float, default=1.0)
parser.add_argument('--chamfer_w', type=float, default=1.0)
parser.add_argument('--emd_w', type=float, default=1.0)

parser.add_argument('--layer', type=int, default=6)
parser.add_argument('--feature', type=list, default=[32,128,512,1024,512,128])
parser.add_argument('--save_dir', type=str, default='./model')
parser.add_argument('--k', type=int, default=20)

args = parser.parse_args()

EPOCH = args.epochs
UNIFORM = args.uniform_w
CHAMFER = args.chamfer_w
EMD = args.emd_w
K = args.k

PATH = f'./k-{K}_putransformer_69000/'

if not os.path.isdir(PATH):
    os.mkdir(PATH)
    print('make directory')

# dataset
loader = DataLoader(args,skip_rate=1)
dataset = torch.utils.data.DataLoader(dataset=loader, batch_size=args.batch_size, shuffle=True,
                num_workers=args.num_workers, pin_memory=True, drop_last=True)

# model Fix ratio & sampling ratio
model = PointUpsamplingTransformer(in_channels=3, out_channels=3, upscale_factor=4).cuda()
model.apply(xavier_init)
count_parameters(model)
# optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
# scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=[int(args.epochs*0.6), int(args.epochs*0.8)], gamma=0.1)
optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
chamfer_dist = ChamferDistance()


if torch.cuda.is_available():
    chamfer_dist.cuda()
    model.cuda()

scaler = amp.GradScaler()

for e in range(EPOCH):

    train_cd_loss = train_cd_loss_rev = train_uniform_loss = train_emd_loss = 0
    SCORE = 1000

    logger = '{}-EPOCH - ChamferDistance : {} - Uniform : {}'
    cnt = 0
    for idx,(sparse_points,dense_points,_) in  enumerate(dataset):
        cnt += 1
        if torch.cuda.is_available():
            sparse_points = sparse_points.cuda()
            dense_points = dense_points.cuda()

        optimizer.zero_grad()

        # with amp.autocast():
        #     dense_output = model(sparse_points)
        #     cd_loss = chamfer_dist(dense_output,dense_points)
        #     cd_loss_rev = chamfer_dist(dense_points,dense_output)
        #     uniform_loss = get_uniform_loss(dense_output)
        #
        #     loss = CHAMFER * 0.5 * (cd_loss + cd_loss_rev) + UNIFORM * uniform_loss
        # scaler.scale(optimizer).backward()
        # scaler.update()

        dense_output = model(sparse_points)

        cd_loss,cd_loss_rev = chamfer_dist(dense_output, dense_points)
        #emd_loss = earth_mover_distance(dense_output.permute(0, 2, 1), dense_points.permute(0, 2, 1))
        cd_loss = torch.mean(cd_loss)
        #emd_loss = torch.mean(emd_loss)
        cd_loss_rev = torch.mean(cd_loss_rev)
        #uniform_loss = get_uniform_loss(dense_output)


        #loss = CHAMFER * 0.5 * (cd_loss + cd_loss_rev) + UNIFORM * uniform_loss + EMD * emd_loss
        loss = 0.5 * (cd_loss + cd_loss_rev)
        loss.backward()
        optimizer.step()

        train_cd_loss += cd_loss.item()
        train_cd_loss_rev += cd_loss_rev.item()
        #train_uniform_loss += uniform_loss.item()
        #train_emd_loss += emd_loss.item()

        if (idx % 10) == 0:

            # print(f'{e + 1} - EPOCH Chamfer Dist : {train_cd_loss/cnt:.6f} Chamfer Dist Rev :'
            #       f' {train_cd_loss_rev/cnt:.6f} Uniform : {train_uniform_loss/cnt:.6f}'
            #       f' EMD : {train_emd_loss/cnt}')
            print(f'{e + 1} - EPOCH Chamfer Dist : {train_cd_loss / cnt:.6f} Chamfer Dist Rev :'
                  f' {train_cd_loss_rev / cnt:.6f}')

    model_params = model.state_dict()
    torch.save(model_params,PATH + f'{e+1}_epoch_model.pth')
    if train_cd_loss < SCORE:
        torch.save(model_params,PATH + 'best_model.pth')
        SCORE = train_cd_loss
