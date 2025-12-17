import os
import torch
import timm
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from pathlib import Path
from tqdm import tqdm
import random
import ipdb
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("--test_our_features", action="store_true", help="Test using our features instead of model features")
parser.add_argument("--ori_dis", type=str, default="Euclidean", choices=["Euclidean", "Cosine"])
parser.add_argument("--sigma", type=float, default=3.0)

args = parser.parse_args()


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

set_seed(42)

root = "OfficeHomeDataset_10072016"  
domains = ["Art", "Clipart", "Product", "Real_World"]
batch_size = 32
device = "cuda" if torch.cuda.is_available() else "cpu"
cache_dir = "officehome_vit_features"
os.makedirs(cache_dir, exist_ok=True)

transform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

print("Loading ViT-B/16 (pretrained on ImageNet) ...")
model = timm.create_model('vit_base_patch16_224', pretrained=True)
model.reset_classifier(0)  
model.eval().to(device)


@torch.no_grad()
def extract_features(dataloader):
    features_list, labels_list = [], []
    for imgs, labels in tqdm(dataloader, desc="Extracting features (GAP)"):
        imgs = imgs.to(device)
        tokens = model.forward_features(imgs)  
        patch_tokens = tokens[:, 1:, :]        
        pooled = patch_tokens.mean(dim=1)     
        features_list.append(pooled.cpu().numpy())
        labels_list.append(labels.numpy())
    return np.concatenate(features_list), np.concatenate(labels_list)


results = {}
for domain in domains:
    print(f"\n--- Processing domain: {domain} ---")
    feat_path = Path(cache_dir) / f"{domain}_features.npy"
    label_path = Path(cache_dir) / f"{domain}_labels.npy"

    if feat_path.exists() and label_path.exists():
        print("Loading cached features...")
        X = np.load(feat_path)
        y = np.load(label_path)

        if args.sigma > 0.0:
            print(f"Adding Gaussian noise with sigma={args.sigma} to features...")
            X = X + np.random.normal(loc=0.0, scale=args.sigma, size=X.shape)


        if args.test_our_features:
            print("using our features.....")
            img = torch.from_numpy(X).to(device)
            img_norm = torch.nn.functional.normalize(img, p=2, dim=1)
        
            if args.ori_dis == "Euclidean":
                img_norm_cos_sim = torch.cdist(img_norm, img_norm, p=2)  
            elif args.ori_dis == "Cosine":
                img_norm_cos_sim = img_norm @ img_norm.T  
            
            X = img_norm_cos_sim.cpu().numpy()
    else:
        print("Extracting and caching features...")
        dataset = datasets.ImageFolder(os.path.join(root, domain), transform=transform)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)
        X, y = extract_features(dataloader)
        np.save(feat_path, X)
        np.save(label_path, y)



    n = len(X)
    idx = np.arange(n)
    np.random.shuffle(idx)
    split = int(0.8 * n)
    train_idx, test_idx = idx[:split], idx[split:]
    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    clf = LogisticRegression(max_iter=1000)
    clf.fit(X_train, y_train)
    acc = accuracy_score(y_test, clf.predict(X_test))
    results[domain] = acc

print(f"Setting: {args}")
print("\n===== Final Accuracy Results =====")
for domain, acc in results.items():
    print(f"{domain}: {acc:.4f}")
