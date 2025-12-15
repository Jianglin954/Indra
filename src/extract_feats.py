import torchvision.datasets as dasets
import torchvision.transforms as transforms
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import ViTImageProcessor, ViTModel, ViTConfig, CLIPProcessor, CLIPModel
from transformers import BertModel, BertTokenizer
from sentence_transformers import SentenceTransformer, models
from transformers import AutoImageProcessor, AutoModel
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
from PIL import Image
import argparse
from torchvision.models.feature_extraction import create_feature_extractor
import timm
import numpy as np
import torch, random
from tqdm import tqdm
import os

import ipdb

COCO_ROOT = "./datasets/MS-COCO/val2017" 
COCO_ANN = "./datasets/MS-COCO/annotations/captions_val2017.json"

# NOCAPS_ROOT = "/shared/group/openimages/validation"
NOCAPS_ROOT = "./datasets/NOCAPS/nocaps_validation_images"
NOCAPS_ANN = "./datasets/NOCAPS/nocaps_val_4500_captions.json"

def parse_args():
    """
    Parse the following arguments for a default parser
    """
    parser = argparse.ArgumentParser(
        description="Running experiments"
    )
    parser.add_argument(
        "--m",
        dest="model_name",
        default="dinov2",
        type=str,
    )
    parser.add_argument(
        "--d",
        dest="dataset",
        default="coco",
        type=str,
    )
    parser.add_argument(
        "--gpu",
        dest="gpu",
        default=1,
        type=int,
    )
    return parser.parse_args()

class FeatureExtractor:
    def __init__(self):
        self.extracted_features = None

    def __call__(self, module, input_, output):
        self.extracted_features = output

def get_model(model_name, device):
    if model_name == "convnext":
        processor = ConvNextImageProcessor.from_pretrained("facebook/convnext-base-224-22k")
        model = ConvNextForImageClassification.from_pretrained("facebook/convnext-base-224-22k").to(device)
        return model.eval(), processor
    elif model_name == "dinov2":
        vision_model_name = "facebook/dinov2-large"
        processor = AutoImageProcessor.from_pretrained(vision_model_name)
        model = AutoModel.from_pretrained(vision_model_name).to(device)
        return model.eval(), processor
    elif model_name == "clip":
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14")
        model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14").to(device)
        return model.eval(), processor
    elif model_name == "allroberta":
        language_model_name = "all-roberta-large-v1"
        language_model = SentenceTransformer(language_model_name).to(device)
        return language_model.eval()
    elif model_name == "bert":
        model_name = "bert-large-uncased"
        tokenizer = BertTokenizer.from_pretrained(model_name)
        model = BertModel.from_pretrained(model_name).to(device)
        return model.eval(), tokenizer
    elif model_name == "vit":
        model = timm.create_model(
            'vit_base_patch16_384.augreg_in1k',
            pretrained=True,
            num_classes=0,  # remove classifier nn.Linear
        ).to(device)
        model = model.eval()
        
        # get model specific transforms (normalization, resize)
        data_config = timm.data.resolve_model_data_config(model)
        transform = timm.data.create_transform(**data_config, is_training=False)
        return model, transform

def get_dataset(dataset):
    if dataset=="coco":
        cap = dasets.CocoCaptions(root = COCO_ROOT,
                        annFile = COCO_ANN)
    elif dataset=="nocaps":
        cap = dasets.CocoCaptions(root = NOCAPS_ROOT,
                        annFile = NOCAPS_ANN)
    return cap 

def run_model(model_name, model_transform, cap, device):
    if model_name == "dinov2":
        model, processor = model_transform
        image_representations = []
    
        for img, target in tqdm(cap):          
            inputs = processor(images=img, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            image_representation = outputs.last_hidden_state[:, 1:, :].mean(dim=1).detach().cpu()[0]    # 去掉 cls token
            # image_representation = outputs.last_hidden_state.mean(dim=1).detach().cpu()[0]  # [1, 257, 1024] 去掉 cls token
            image_representations.append(image_representation)      
        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'./features/{dataset}_{model_name}_img.pt')
    elif model_name == "vit":
        model, transform = model_transform
        image_representations = []
    
        for img, target in tqdm(cap):
            input = transform(img).unsqueeze(0).to(device)
            with torch.no_grad():
                output = model(input)[0]  # output is (batch_size, num_features) shaped tensor
            image_representations.append(output)
        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'./features/{dataset}_{model_name}_img.pt')
    elif model_name == "convnext":
        model, processor = model_transform
        image_representations = []
    
        for img, target in tqdm(cap):
            
            inputs = processor(images=img, return_tensors="pt")
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model.convnext(**inputs)
            #print(outputs.last_hidden_state.shape)
            image_representation = outputs.last_hidden_state.reshape(1, 1024, -1).mean(2).detach().cpu()[0]
            image_representation = image_representation / np.linalg.norm(image_representation, axis=0, keepdims=True)
            image_representations.append(image_representation)

        image_representations_tensor = torch.stack(image_representations)
        torch.save(image_representations_tensor, f'./features/{dataset}_{model_name}_img.pt')
    
    elif model_name == "clip":
        model, processor = model_transform
        image_representations = []
        text_representations = []
        
        for img, target in tqdm(cap):
            inputs = processor(text=target, images=img, return_tensors="pt", padding=True)
            inputs = inputs.to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            text_representation = outputs.text_embeds.detach().cpu().squeeze()
            if text_representation.shape[0] > 5:
                text_representation = text_representation[:5]
            # text_representation = text_representation.mean(dim=0).squeeze()
            image_representation = outputs.image_embeds.detach().cpu().squeeze()
        
            text_representations.append(text_representation)
            image_representations.append(image_representation)

        text_representations_tensor = torch.stack(text_representations)
        image_representations_tensor = torch.stack(image_representations)
        
        text = text_representations_tensor.detach().clone().reshape(-1, text_representations_tensor.shape[2]) 
        img = image_representations_tensor.detach().clone()
        
        text_norm = torch.nn.functional.normalize(text, p=2, dim=1) # L2 normalization on feature dimension
        img_norm = torch.nn.functional.normalize(img, p=2, dim=1)
        t2i_att = text_norm @ img_norm.T
        i2t_att = img_norm @ text_norm.T
        
        torch.save(text_representations_tensor, f'./features/{dataset}_{model_name}_text.pt')
        torch.save(image_representations_tensor, f'./features/{dataset}_{model_name}_img.pt')
        torch.save(t2i_att, f'./features/{dataset}_{model_name}_t2i_att.pt')
        torch.save(i2t_att, f'./features/{dataset}_{model_name}_i2t_att.pt')
    
    elif model_name == "bert":
        model, tokenizer = model_transform
        text_representations = []
        for img, target in tqdm(cap):
            tokenized_inputs = tokenizer(target, padding="max_length",  truncation=True, max_length=512, add_special_tokens=False, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model(**tokenized_inputs)                 
            # output.last_hidden_state  # [5, 512, 1024]
            attention_mask = tokenized_inputs["attention_mask"].unsqueeze(-1)  # [5, 512, 1]
            token_embeddings = (output.last_hidden_state * attention_mask).sum(dim=1) / attention_mask.sum(dim=1)
            text_representation = torch.Tensor(token_embeddings.mean(dim=0))    # average 5 sentences
            text_representations.append(text_representation)
            
        text_representations_tensor = torch.stack(text_representations)
        torch.save(text_representations_tensor, f'./features/{dataset}_{model_name}_text.pt')
        
    elif model_name == "allroberta":
        language_model = model_transform
        text_representations = []

        for img, target in tqdm(cap):
            with torch.no_grad():
                output = language_model.encode(target)                  # (5, 1024)
            text_representation = torch.Tensor(output)
            text_representation = text_representation.mean(dim=0)
            text_representations.append(text_representation)
            
        text_representations_tensor = torch.stack(text_representations)
        torch.save(text_representations_tensor, f'./features/{dataset}_{model_name}_text.pt')

        
if __name__ == "__main__":
    args = parse_args()

    gpu = f'cuda:{args.gpu}'
    device = torch.device(gpu)
    torch.cuda.empty_cache()

    model_name = args.model_name
    dataset = args.dataset
    
    model_transform = get_model(model_name, device)
    cap = get_dataset(dataset)
    run_model(model_name, model_transform, cap, device)