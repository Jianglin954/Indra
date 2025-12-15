import argparse
import torch
import numpy as np
import timm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import random
import os
import ipdb
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent  # project_root
DATASETS = ROOT / "datasets"
FEATURES = ROOT / "features"



def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

@torch.no_grad()
def extract_features(model, dataloader, device, save_path_prefix):
    feats, labels = [], []
    for imgs, lbls in tqdm(dataloader, desc=f"Extracting: {save_path_prefix}"):
        imgs = imgs.to(device)
        f = model(imgs)
        feats.append(f.cpu().numpy())
        labels.append(lbls.numpy())
    X = np.concatenate(feats)
    y = np.concatenate(labels)
    np.save(f"{save_path_prefix}_X.npy", X)
    np.save(f"{save_path_prefix}_y.npy", y)
    return X, y

def load_or_extract(model, loader, device, prefix, use_cache):
    if use_cache and os.path.exists(f"{prefix}_X.npy") and os.path.exists(f"{prefix}_y.npy"):
        X = np.load(f"{prefix}_X.npy")
        y = np.load(f"{prefix}_y.npy")

    else:
        X, y = extract_features(model, loader, device, prefix)
    return X, y

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", type=str, default="dinov2",
                        choices=["dinov2", "resnet50", "vit_base_patch16_224", "convnext_base"])
    parser.add_argument("--dataset", type=str, default="cifar10", choices=["SVHN", "cifar10", "cifar100"])
    parser.add_argument("--use_cache", action="store_true", help="Use saved .npy features if available")
    
    parser.add_argument("--test_our_features", action="store_true", help="Test using our features instead of model features")

    parser.add_argument("--ori_dis", type=str, default="Euclidean", choices=["Euclidean", "Cosine"])
    
    parser.add_argument("--sigma", type=float, default=3.0)

    args = parser.parse_args()
    
    
    data_root = DATASETS / "CIFAR"
    set_seed()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    if args.dataset == "SVHN":
        DatasetClass = datasets.SVHN
        train_dataset = DatasetClass(root=data_root, split='train', download=True, transform=transform)
        test_dataset = DatasetClass(root=data_root, split='test', download=True, transform=transform)
    else:
        DatasetClass = datasets.CIFAR10 if args.dataset == "cifar10" else datasets.CIFAR100
        train_dataset = DatasetClass(root=data_root, train=True, download=True, transform=transform)
        test_dataset = DatasetClass(root=data_root, train=False, download=True, transform=transform)
        
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=2)


    if args.backbone == "dinov2":
        model = torch.hub.load('facebookresearch/dinov2', 'dinov2_vitl14')
    else:
        model = timm.create_model(args.backbone, pretrained=True, num_classes=0)
    model.eval().to(device)


    prefix_train = f"{FEATURES}/{args.dataset}_{args.backbone}_train"
    prefix_test = f"{FEATURES}/{args.dataset}_{args.backbone}_test"
    X_train, y_train = load_or_extract(model, train_loader, device, prefix_train, args.use_cache)
    X_test, y_test = load_or_extract(model, test_loader, device, prefix_test, args.use_cache)


    if args.sigma > 0.0:
        print(f"Adding Gaussian noise with sigma={args.sigma} to features...")
        X_train = X_train + np.random.normal(loc=0.0, scale=args.sigma, size=X_train.shape)
        X_test  = X_test  + np.random.normal(loc=0.0, scale=args.sigma, size=X_test.shape)

    
    if args.test_our_features:
        print("using our features.....")

        img = torch.cat([torch.from_numpy(X_train), torch.from_numpy(X_test)], dim=0)
        img_norm = torch.nn.functional.normalize(img, p=2, dim=1)

    
        if args.ori_dis == "Euclidean":
            print("Using Euclidean distance...")
            img_norm_cos_sim = torch.cdist(img_norm, img_norm, p=2)  # (N, D) @ (D, N) -> (N, N)
        elif args.ori_dis == "Cosine":
            print("Using Cosine similarity...")
            img_norm_cos_sim = img_norm @ img_norm.T  # (N, D) @ (D, N) -> (N, N)
        

        n_train = X_train.shape[0]
        n_test  = X_test.shape[0]
        X_train = img_norm_cos_sim[:n_train, :]  # [n_train, n_total]
        X_test  = img_norm_cos_sim[n_train:, :]  # [n_test,  n_total]   
        
        X_train = X_train.cpu().numpy()
        X_test  = X_test.cpu().numpy() 


    clf = LogisticRegression(max_iter=100, multi_class="multinomial", solver="lbfgs")
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    print(f"\n Accuracy on {args.dataset.upper()} using {args.backbone}, {args.test_our_features}, {args.ori_dis} sigma_{args.sigma}: {acc:.4f}")

if __name__ == "__main__":
    main()