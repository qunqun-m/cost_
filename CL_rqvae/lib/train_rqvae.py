import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch.nn.functional as F
import collections
from optimizers import AdamOptimizer
from optimizers.lr_schedulers import InverseSquareRootSchedule
from sklearn.metrics.pairwise import euclidean_distances
import torch.nn as nn
from lib.t5_modules import Encoder, Decoder, AverageMeter, T5Stream
import torch
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import logging

loss_meter = collections.defaultdict(lambda: AverageMeter())

def train_rqvae(data_set_name, rqvae_model_name, final_feat, rqvae_config, device):
    num_quantizers, codebook_size = rqvae_config['num_quantizers'], rqvae_config['codebook_size']
    C, D, batch_size = rqvae_config['C'], rqvae_config['D'], rqvae_config['batch_size']
    cos_add, total_epoch =  rqvae_config['cos_add'], rqvae_config['total_epoch']
    distill, mse = rqvae_config['distill'], rqvae_config['mse']
    
    args = {}
    args["lr"] = 1e-3 # 1e-5 #
    args["weight_decay"] = 1e-7
    args["warmup_updates"] = 3000 #20000 1000 #
    args["warmup_init_lr"] = 1e-7

    data = final_feat
    if rqvae_model_name == 'rq':
        model = T5Stream(C=C, D=D, num_quantizers=num_quantizers, codebook_size=codebook_size).to(device)

    parameters = list(model.parameters())
    optimizer = AdamOptimizer(args, parameters)
    lr_scheduler = InverseSquareRootSchedule(args, optimizer)

    num_update = 0
    item_num = len(data)
    model.train()

    codes = torch.zeros(len(data), num_quantizers).to(device)
    for epoch in range(1, total_epoch+1): # 20000 training
        def print_log():
            msg = 'Epoch {}, lr = {:.7f}, '.format(epoch, curr_lr)

            # msg = 'Epoch {}, Batch {},  '.format(epoch, bid)
            for k, v in loss_meter.items():
                msg += '{} = {:.6f}, '.format(k, v.avg)
                v.reset()
            print(msg)
            # logging.info(msg)
        
        def embtopk(input, out):
            input = input.cpu()
            out = out.detach().cpu()
            K1= 20
            K2=100
            dist1 = euclidean_distances(input, input)
            dist2 = euclidean_distances(out, out)
            near1 = np.argsort(dist1, axis=1)[:,:K1]
            near2 = np.argsort(dist2, axis=1)[:,:K1]
            near1_ = np.argsort(dist1, axis=1)[:,:K2]
            near2_ = np.argsort(dist2, axis=1)[:,:K2]

            precision1 = 0.0
            precision2 = 0.0
            for i in range(near1.shape[0]):
                precision1 += len(set(near1[i]).intersection(set(near2[i])))/K1
                precision2 += len(set(near1_[i]).intersection(set(near2_[i])))/K2
            precision1 = precision1/near1.shape[0]
            precision2 = precision2/near1_.shape[0]
            return precision1, precision2

        start_id = 0
        model.train()
        while start_id < item_num:
            model.zero_grad()
            input = torch.from_numpy(data[start_id:start_id+batch_size]).to(device).float()
            # input = F.normalize(input, dim=-1)
            out, indices, commit_loss = model(input)
            # out = F.normalize(out, dim=-1)

            prec1, prec2 = embtopk(input, out)
            loss_meter['prec@20'].update(prec1)
            loss_meter['prec@100'].update(prec2)

            # loss_recon = F.mse_loss(out, input, reduction='mean')#.sum(dim=-1).mean()
            loss_commit = commit_loss.mean()
            loss_meter['loss_com'].update(loss_commit.item())
            
            loss = loss_commit * 0.25

            if mse:
                loss_recon = (input - out).abs().mean()
                loss_meter['loss_recon'].update(loss_recon.item())
                loss += loss_recon
            
            if distill:
                T =  0.1 # 0.5 # 1.0 #100.0 #
                input = F.normalize(input, dim=1)
                out = F.normalize(out, dim=1)
                mmdist = torch.matmul(input, out.t())/T

                pos = torch.exp(mmdist.diagonal())
                neg = torch.exp(mmdist).sum(dim = 1)
                # print(pos[0],neg[0])
                loss_cd = pos / neg
                loss_cd = torch.log(loss_cd+1e-8)
                loss_cd = - torch.mean(loss_cd)
                loss_meter['loss_cd'].update(loss_cd.item())
                if mse:
                    loss = loss + loss_cd*0.1
                else:
                    loss = loss + loss_cd

            if cos_add:
                cos_dis = cosine_similarity(input.cpu().detach().numpy(), out.cpu().detach().numpy())
                diag_matrix = cos_dis.diagonal()
                cos = np.mean(diag_matrix)
                loss_meter['cos'].update(cos.item())

            optimizer.zero_grad()
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)

            optimizer.step()
            num_update += 1
            curr_lr = lr_scheduler.step_update(num_update)

            if start_id + batch_size < item_num:
                codes[start_id:start_id+batch_size] = indices
            else:
                codes[start_id: ] = indices

            start_id=start_id+batch_size
            #
            bid = (start_id // batch_size)
            if bid % 10 == 0:
                print_log()

        if epoch % 10 == 0:
            state_dict = {
                'num_updates': num_update,
                'config': args,
                'model_parameters': model.state_dict(),
                'codes': codes
            }
            path = 'data/{}/{}_{}_mse{}_distill{}.pth'.format(data_set_name, rqvae_model_name, epoch, mse, distill)
            torch.save(state_dict, path)
            x = 'save model to {}, num_updates {}.'.format(path, num_update)
            print(x)

    print("over")

    def collision(codes):
        dict = {}
        collid = set()
        k = codebook_size
        def unique_code(code, k):
            lencode = len(code)
            unique = 0.0
            for i in range(lencode):
                unique += code[i] * k **(lencode -i-1)
            return unique

        for i in range(len(codes)):
            uni = unique_code(codes[i], k)
            if uni in dict.keys():
                collid.add(i)
            dict[i] = 0
        print("collision:{}, total:{}".format(len(collid), len(codes)))

    collision(codes)
    return codes