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

DATA_LOCATION = "./features/audio"
AVG_RESULT_LOCATION = "./results"

import ipdb

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
        default="clap-larger-general",
        type=str,
    )
    parser.add_argument(
        "--text_model",
        dest="text_model",
        help="text_model",
        default="clap-larger-general",
        type=str,
    )
    parser.add_argument(
        "--base_data",
        dest="base_data",
        help="base_data",
        default="TIMIT",
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

if __name__ == "__main__":
    args = parse_args()
    
    print("---------------------------------")
    print(args)
    print(args.vid_model, args.text_model)
    
    os.environ["CUDA_VISIBLE_DEVICES"]=str(args.gpu)
    gpu = 'cuda'
    device = torch.device(gpu)

    target_model = args.vid_model
    source_model = args.text_model
    base_data = args.base_data
    query_data = args.query_data
    
    kkk = args.kkk
    ori_dis = args.dist
    
    clap_source = torch.load(f"{DATA_LOCATION}/{base_data}_clap-larger-general_text.pt", map_location=gpu).to(device)
    clap_target = torch.load(f"{DATA_LOCATION}/{base_data}_clap-larger-general_audio.pt", map_location=gpu).to(device)
    

    clap_text = clap_source.detach().clone()
    clap_audio = clap_target.detach().clone()
    clap_text_norm = torch.nn.functional.normalize(clap_text, p=2, dim=1) # L2 normalization on feature dimension
    clap_audio_norm = torch.nn.functional.normalize(clap_audio, p=2, dim=1)
    
    t2i_atten_clip = clap_text_norm @ clap_audio_norm.T
    i2t_atten_clip = clap_audio_norm @ clap_text_norm.T
    
    
    
    source_base_large = torch.load(f"{DATA_LOCATION}/{base_data}_{source_model}_text.pt", map_location=gpu).to(device)
    target_base_large = torch.load(f"{DATA_LOCATION}/{base_data}_{target_model}_audio.pt", map_location=gpu).to(device)
    
    
    text = source_base_large.detach().clone()
    img = target_base_large.detach().clone()
    
    
    if text.shape[1] > img.shape[1]:
        print(f"text dimension {text.shape[1]} > img dimension {img.shape[1]}! Using PCA for dimension reduction!")
        pca = PCA(n_components=img.shape[1])   
        text_reduced = pca.fit_transform(text.cpu().numpy()) 
        text = torch.tensor(text_reduced).to(device)
    elif text.shape[1] < img.shape[1]:
        print(f"img dimension {img.shape[1]} > text dimension {text.shape[1]}! Using PCA for dimension reduction!")
        pca = PCA(n_components=text.shape[1])    
        img_reduced = pca.fit_transform(img.cpu().numpy())  
        img = torch.tensor(img_reduced).to(device)
    else:
        print(f"text dimension {text.shape[1]} == img dimension {img.shape[1]}!")
        
    
    text_norm = torch.nn.functional.normalize(text, p=2, dim=1)
    img_norm = torch.nn.functional.normalize(img, p=2, dim=1)

    ### original features 
    if ori_dis == "Euclidean":
        t2i_cosim_ori = torch.cdist(text_norm, img_norm, p=2)
    elif ori_dis == "Cosine":
        t2i_cosim_ori = text_norm @ img_norm.T
    topk_val_ori, topk_ind_ori = torch.topk(t2i_cosim_ori, kkk, dim=1)
    t2i_sim_ori = torch.gather(t2i_atten_clip, dim=1, index=topk_ind_ori)
    t2i_ori_score = t2i_sim_ori.sum() / t2i_sim_ori.shape[0]   
    
    if ori_dis == "Euclidean":
        i2t_cosim_ori = torch.cdist(img_norm, text_norm, p=2)
    elif ori_dis == "Cosine":
        i2t_cosim_ori = img_norm @ text_norm.T
    topk_val_ori, topk_ind_ori = torch.topk(i2t_cosim_ori, kkk, dim=1)
    i2t_sim_ori = torch.gather(i2t_atten_clip, dim=1, index=topk_ind_ori)
    i2t_ori_score = i2t_sim_ori.sum() / i2t_sim_ori.shape[0]    
    
    
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
    t2i_att_score = t2i_sim_att.sum() / t2i_sim_att.shape[0]        

    i2t_dist_atten = torch.cdist(img_norm_cos_sim, text_norm_cos_sim, p=2)
    topk_val_att, topk_ind_att = torch.topk(i2t_dist_atten, kkk, dim=1, largest=False)
    i2t_sim_att = torch.gather(i2t_atten_clip, dim=1, index=topk_ind_att)
    i2t_att_score = i2t_sim_att.sum() / i2t_sim_att.shape[0]        
    
    
    ### clip features
    t2i_topk_val_clip, topk_ind_clip = torch.topk(t2i_atten_clip, kkk, dim=1)
    t2i_clip_score = t2i_topk_val_clip.sum() / t2i_topk_val_clip.shape[0]   
    
    i2t_topk_val_clip, topk_ind_clip = torch.topk(i2t_atten_clip, kkk, dim=1)
    i2t_clip_score = i2t_topk_val_clip.sum() / i2t_topk_val_clip.shape[0]   
    
    
    print("t2i_ori_score", t2i_ori_score)
    print("i2t_ori_score", i2t_ori_score)
    
    print("t2i_att_score", t2i_att_score)
    print("i2t_att_score", i2t_att_score)
    
    print("t2i_clip_score", t2i_clip_score)
    print("i2t_clip_score", i2t_clip_score)
    
    exit()


