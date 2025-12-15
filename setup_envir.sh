conda create -yn indra python=3.10
conda run -n indra pip install numpy
conda run -n indra pip install scipy
conda run -n indra pip install scikit-learn
conda run -n indra pip install tqdm
conda run -n indra pip install transformers
conda run -n indra pip install sentence-transformers
conda run -n indra pip install pillow
conda run -n indra pip install requests
conda run -n indra pip install timm
conda run -n indra pip install pandas
conda run -n indra pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu128
conda run -n indra pip install pycocotools  # MSCOCO
conda init
conda activate indra