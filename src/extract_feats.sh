

####  coco 
## clip features
python ./src/extract_feats.py --m clip       --d coco --gpu 0        # image & text embedding 
## image
python ./src/extract_feats.py --m vit        --d coco --gpu 0        # image embedding 
python ./src/extract_feats.py --m convnext   --d coco --gpu 0        # image embedding 
python ./src/extract_feats.py --m dinov2     --d coco --gpu 0        # image embedding 
## text
python ./src/extract_feats.py --m bert       --d coco --gpu 0        # text embedding
python ./src/extract_feats.py --m allroberta --d coco --gpu 0        # text embedding


####  nocaps
## clip features
python ./src/extract_feats.py --m clip       --d nocaps --gpu 0
## image
python ./src/extract_feats.py --m vit        --d nocaps --gpu 0
python ./src/extract_feats.py --m convnext   --d nocaps --gpu 0       
python ./src/extract_feats.py --m dinov2     --d nocaps --gpu 0  
## text
python ./src/extract_feats.py --m bert       --d nocaps --gpu 0
python ./src/extract_feats.py --m allroberta --d nocaps --gpu 0