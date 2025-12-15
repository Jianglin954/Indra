import numpy as np
from sklearn import datasets, cluster
import torch
import os
from tqdm import tqdm
from sklearn.decomposition import PCA
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
import pandas as pd

import argparse
from datetime import datetime
from sklearn.cross_decomposition import CCA


DATA_LOCATION = "./features"

import ipdb
from prompt_toolkit.history import InMemoryHistory
history = InMemoryHistory() 

from sklearn.decomposition import PCA

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Running experiments"
    ) 
    parser.add_argument(
        '--dist', 
        dest="dist",
        type=str, 
        default='Cosine', 
        help='[Euclidean, Cosine] used for self-attention calculation',
    )
    
    parser.add_argument(
        "--vid_model",
        dest="vid_model",
        help="vid_model",
        default="dinov2",
        type=str,
    )
    parser.add_argument(
        "--text_model",
        dest="text_model",
        help="text_model",
        default="allroberta",
        type=str,
    )
    parser.add_argument(
        "--base_data",
        dest="base_data",
        help="base_data",
        default="coco",
        type=str,
    )
    parser.add_argument(
        "--query_data",
        dest="query_data",
        help="query data",
        default="nocaps",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        help="gpu",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--kkk",
        dest="kkk",
        help="retrieve/match top-kkk sample",
        default=0,
        type=int,
    )
    
    return parser.parse_args()






import torch

def cca_torch(X, Y, outdim_size, epsilon=1e-10):
    """
    X: [n, dx] - first view
    Y: [n, dy] - second view
    outdim_size: number of canonical components
    """
    device = X.device
    X -= X.mean(dim=0)
    Y -= Y.mean(dim=0)

    n = X.shape[0]

    # Covariance matrices
    SigmaXX = X.T @ X / (n - 1) + epsilon * torch.eye(X.shape[1], device=device)
    SigmaYY = Y.T @ Y / (n - 1) + epsilon * torch.eye(Y.shape[1], device=device)
    SigmaXY = X.T @ Y / (n - 1)

    # Inverse square root via eigen-decomposition
    def inv_sqrtm(Sigma):
        D, V = torch.linalg.eigh(Sigma)
        D_inv_sqrt = torch.diag(torch.clamp(D, min=epsilon).pow(-0.5))
        return V @ D_inv_sqrt @ V.T

    SigmaXX_inv_sqrt = inv_sqrtm(SigmaXX)
    SigmaYY_inv_sqrt = inv_sqrtm(SigmaYY)

    # T matrix
    T = SigmaXX_inv_sqrt @ SigmaXY @ SigmaYY_inv_sqrt

    # SVD on T
    U, S, Vh = torch.linalg.svd(T)

    Wx = SigmaXX_inv_sqrt @ U[:, :outdim_size]  # Projection for X
    Wy = SigmaYY_inv_sqrt @ Vh.T[:, :outdim_size]  # Projection for Y

    return Wx, Wy, S[:outdim_size]  # canonical correlations






if __name__ == "__main__":
    args = parse_args()
    
    print("---------------------------------")
    print(args)
    print(args.vid_model, args.text_model)
    
    #torch.cuda.empty_cache()
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    gpu = 'cuda'
    device = torch.device(gpu)

    target_model = args.vid_model
    source_model = args.text_model
    base_data = args.base_data
    query_data = args.query_data
    
    kkk = args.kkk
    ori_dis = args.dist
    
    source_base_large = torch.load(f"{DATA_LOCATION}/{base_data}_{source_model}_text.pt", map_location=gpu).to(device)
    if target_model == "linear_proj_ICLR23":
        target_base_large = torch.load(f"{DATA_LOCATION}/{base_data}_{target_model}_img_with_proj.pt", map_location=gpu).to(device)
    else:
        target_base_large = torch.load(f"{DATA_LOCATION}/{base_data}_{target_model}_img.pt", map_location=gpu).to(device)
    t2i_atten_clip = torch.load(f"{DATA_LOCATION}/{base_data}_clip_t2i_att.pt", map_location=gpu).to(device)
    # i2t_atten_clip = torch.load(f"{DATA_LOCATION}/{base_data}_clip_i2t_att.pt", map_location=gpu).to(device)
    i2t_atten_clip = t2i_atten_clip.T
    
    text = source_base_large.detach().clone()
    img = target_base_large.detach().clone()
    
    text_cop = source_base_large.detach().clone()
    img_cop = target_base_large.detach().clone()
    # ipdb.set_trace()
    
    if text.shape[1] > img.shape[1]:
        print(f"text dimension {text.shape[1]} > img dimension {img.shape[1]}! Using PCA for dimension reduction!")
        pca = PCA(n_components=img.shape[1])    # reduce to 1024 dimention
        text_reduced = pca.fit_transform(text.cpu().numpy())  # Shape: (n, 2048)
        text = torch.tensor(text_reduced).to(device)
    elif text.shape[1] < img.shape[1]:
        print(f"img dimension {img.shape[1]} > text dimension {text.shape[1]}! Using PCA for dimension reduction!")
        pca = PCA(n_components=text.shape[1])    # reduce to 1024 dimention
        img_reduced = pca.fit_transform(img.cpu().numpy())  # Shape: (n, 2048)
        img = torch.tensor(img_reduced).to(device)
    else:
        print(f"text dimension {text.shape[1]} == img dimension {img.shape[1]}!")
        
    
            
    ### similarity using original features 
    text_norm = torch.nn.functional.normalize(text, p=2, dim=1) # L2 normalization on feature dimension
    img_norm = torch.nn.functional.normalize(img, p=2, dim=1)
    
    text_norm_cop = torch.nn.functional.normalize(text_cop, p=2, dim=1) # L2 normalization on feature dimension
    img_norm_cop = torch.nn.functional.normalize(img_cop, p=2, dim=1)
    
    text_norm = text_norm.to(torch.float32)
    img_norm = img_norm.to(torch.float32)


    ### original features 
    if ori_dis == "Euclidean":
        t2i_cosim_ori = torch.cdist(text_norm, img_norm, p=2)
    elif ori_dis == "Cosine":
        t2i_cosim_ori = text_norm @ img_norm.T
    topk_val_ori, topk_ind_ori = torch.topk(t2i_cosim_ori, kkk, dim=1)
    t2i_sim_ori = torch.gather(t2i_atten_clip, dim=1, index=topk_ind_ori)
    t2i_ori_score = t2i_sim_ori.sum() / t2i_sim_ori.shape[0]    # 0.9450
    
    if ori_dis == "Euclidean":
        i2t_cosim_ori = torch.cdist(img_norm, text_norm, p=2)
    elif ori_dis == "Cosine":
        i2t_cosim_ori = img_norm @ text_norm.T
    topk_val_ori, topk_ind_ori = torch.topk(i2t_cosim_ori, kkk, dim=1)
    i2t_sim_ori = torch.gather(i2t_atten_clip, dim=1, index=topk_ind_ori)
    i2t_ori_score = i2t_sim_ori.sum() / i2t_sim_ori.shape[0]    # 0.9828
    
    

    ### CCA
    #XX, YY = img_norm.clone().cpu().numpy(), text_norm.clone().cpu().numpy()
    #cca = CCA(n_components=500)
    #X_c, Y_c = cca.fit_transform(XX, YY)

    X_c, Y_c, corr = cca_torch(img_norm_cop, text_norm_cop, outdim_size=500)

    if ori_dis == "Euclidean":
        t2i_cca = torch.cdist(Y_c, X_c, p=2)
    elif ori_dis == "Cosine":
        t2i_cca = Y_c @ X_c.T
    topk_val_cca, topk_ind_cca = torch.topk(t2i_cca, kkk, dim=1)
    t2i_sim_cca = torch.gather(t2i_atten_clip, dim=1, index=topk_ind_cca)
    t2i_ori_cca = t2i_sim_cca.sum() / t2i_sim_cca.shape[0]    # 0.9450
    
    if ori_dis == "Euclidean":
        i2t_cca = torch.cdist(X_c, Y_c, p=2)
    elif ori_dis == "Cosine":
        i2t_cca = X_c @ Y_c.T
    topk_val_cca2, topk_ind_cca2 = torch.topk(i2t_cca, kkk, dim=1)
    i2t_sim_cca = torch.gather(i2t_atten_clip, dim=1, index=topk_ind_cca2)
    i2t_ori_cca = i2t_sim_cca.sum() / i2t_sim_cca.shape[0]    # 0.9828

    print(torch.norm(t2i_cca - i2t_cca.T, p=2) )
    print(torch.allclose(t2i_cca, i2t_cca.T, rtol=1e-5, atol=1e-8))


    ### attention scores as features 
    if ori_dis == "Euclidean":
        text_norm_cos_sim = torch.cdist(text_norm, text_norm, p=2)  # (N, D) @ (D, N) -> (N, N)
        img_norm_cos_sim = torch.cdist(img_norm, img_norm, p=2)  # (N, D) @ (D, N) -> (N, N)
    elif ori_dis == "Cosine":
        text_norm_cos_sim = text_norm @ text_norm.T  # (N, D) @ (D, N) -> (N, N)
        img_norm_cos_sim = img_norm @ img_norm.T  # (N, D) @ (D, N) -> (N, N)
    

    t2i_dist_atten = torch.cdist(text_norm_cos_sim, img_norm_cos_sim, p=2)
    topk_val_att, topk_ind_att = torch.topk(t2i_dist_atten, kkk, dim=1, largest=False)
    t2i_sim_att = torch.gather(t2i_atten_clip, dim=1, index=topk_ind_att)
    t2i_att_score = t2i_sim_att.sum() / t2i_sim_att.shape[0]        # 1.9781


    i2t_dist_atten = torch.cdist(img_norm_cos_sim, text_norm_cos_sim, p=2)
    topk_val_att, topk_ind_att = torch.topk(i2t_dist_atten, kkk, dim=1, largest=False)
    i2t_sim_att = torch.gather(i2t_atten_clip, dim=1, index=topk_ind_att)
    i2t_att_score = i2t_sim_att.sum() / i2t_sim_att.shape[0]        # 1.8635
    
    
    ### clip features
    t2i_topk_val_clip, topk_ind_clip = torch.topk(t2i_atten_clip, kkk, dim=1)
    t2i_clip_score = t2i_topk_val_clip.sum() / t2i_topk_val_clip.shape[0]   # 2.7341
    
    i2t_topk_val_clip, topk_ind_clip = torch.topk(i2t_atten_clip, kkk, dim=1)
    i2t_clip_score = i2t_topk_val_clip.sum() / i2t_topk_val_clip.shape[0]   # 2.6610
    
    
    print("__________orig ________________")
    print("t2i_ori_score", t2i_ori_score)
    print("i2t_ori_score", i2t_ori_score)
    
    print("__________cca ________________")
    print("t2i_ori_cca", t2i_ori_cca)
    print("i2t_ori_cca", i2t_ori_cca)

    print("__________ours ________________")
    print("t2i_att_score", t2i_att_score)
    print("i2t_att_score", i2t_att_score)
    
    print("__________clip ________________")
    print("t2i_clip_score", t2i_clip_score)
    print("i2t_clip_score", i2t_clip_score)
    
    exit()


