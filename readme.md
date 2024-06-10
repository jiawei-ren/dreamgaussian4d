<div align="center">

<h1>DreamGaussian4D:<br>Generative 4D Gaussian Splatting</h1>

<div>
Jiawei Ren<sup>*</sup>&emsp;Liang Pan<sup>*</sup>&emsp;Jiaxiang Tang&emsp;Chi Zhang&emsp;Ang Cao&emsp;Gang Zeng&emsp;Ziwei Liu<sup>&dagger;</sup>
</div>
<div>
    S-Lab, Nanyang Technological University&emsp;
    Shanghai AI Laboratory&emsp;<br>
    Peking University &emsp;
    University of Michigan &emsp;<br>
    <sup>*</sup>equal contribution <br>
    <sup>&dagger;</sup>corresponding author 
</div>


<div>
   <strong>Arxiv 2023</strong>
</div>

<div>
<a target="_blank" href="https://arxiv.org/abs/2312.17142">
  <img src="https://img.shields.io/badge/arXiv-2312.17142-b31b1b.svg" alt="arXiv Paper"/>
</a>
<a href="https://hits.seeyoufarm.com"><img src="https://hits.seeyoufarm.com/api/count/incr/badge.svg?url=https%3A%2F%2Fgithub.com%2Fjiawei-ren%2Fdreamgaussian4d&count_bg=%2379C83D&title_bg=%23555555&icon=&icon_color=%23E7E7E7&title=hits&edge_flat=false"/></a>
</div>





https://github.com/jiawei-ren/dreamgaussian4d/assets/72253125/8fdadc58-1ad8-4664-a6f8-70e20c612c10







---

<h4 align="center">
  <a href="https://jiawei-ren.github.io/projects/dreamgaussian4d/" target='_blank'>[Project Page]</a> |
  <a href="https://arxiv.org/abs/2312.17142" target='_blank'>[Paper]
  </a> | 
  <a href="https://huggingface.co/spaces/jiawei011/dreamgaussian4d"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>

</h4>

</div>

### News
- 2024.6.10: add gradio demo. <a href="https://huggingface.co/spaces/jiawei011/dreamgaussian4d"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Gradio%20Demo-Huggingface-orange"></a>

- 2024.6.9: 
  - support [LGM](https://github.com/3DTopia/LGM) for static 3D generation.
  - add support for video-to-4d generation and evaluation scripts for the [Consistent4D](https://consistent4d.github.io/) benchmark. Results are in our updated [project page](https://jiawei-ren.github.io/projects/dreamgaussian4d/) and [report](https://arxiv.org/abs/2312.17142).
  - improve the implementation for better speed and quality. Add a gradio demo for image-to-4d. 

## Install
```bash
# python 3.10 cuda 11.8 
conda create -n dg4d python=3.10 -y && conda activate dg4d
conda install -c "nvidia/label/cuda-11.8.0" cuda-toolkit

pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip install xformers==0.0.23 --no-deps --index-url https://download.pytorch.org/whl/cu118

# other dependencies
pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# for mesh extraction
pip install git+https://github.com/NVlabs/nvdiffrast/
```

To use pretrained LGM:

```bash
# for LGM
mkdir pretrained && cd pretrained
wget https://huggingface.co/ashawkey/LGM/resolve/main/model_fp16_fixrot.safetensors
cd ..
```



## Image-to-4D
##### (Optional) Preprocess input image
```bash
python scripts/process.py data/anya.png
```
##### Step 1: Generate driving videos
```bash
python scripts/gen_vid.py --path data/anya_rgba.png --seed 42 --bg white
```
##### Step 2: static generation
Static generation with [LGM](https://github.com/3DTopia/LGM):
```bash
python lgm/infer.py big --test_path data/anya_rgba.png
```
Optionally, we support static generation with [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian):
```bash
python dg.py --config configs/dg.yaml input=data/anya_rgba.png
```
See `configs/dghd.yaml` for high-quality DreamGaussian training configurations.

##### Step 3: dynamic generation
```bash
# load static 3D from LGM
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png

# (Optional) to load static 3D from DreamGaussian, add `radius=2`
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png radius=2

# (Optional) to turn on viser GUI, add `gui=True`, e.g.:
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png gui=True
```
See `configs/4d_low.yaml` and `configs/4d_demo.yaml` for more memory-friendly and faster optimization configurations.

##### (Optional) Step 4: mesh refinment
```bash
# export mesh after temporal optimization by adding `mesh_format=obj`
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png mesh_format=obj

# mesh refinement
python main2_4d.py --config configs/refine.yaml input=data/anya_rgba.png

# (Optional) to load static 3D from DreamGaussian, add `radius=2`
python main2_4d.py --config configs/refine.yaml input=data/anya_rgba.png radius=2
```

## Video-to-4D
##### Prepare Data
Download [Consistent4D data](https://consistent4d.github.io/) to `data/CONSISTENT4D_DATA`. `python scripts/add_bg_to_gt.py` will add white background to ground-truth novel views.

##### Step 1: static generation
```bash
python lgm/infer.py big --test_path data/CONSISTENT4D_DATA/in-the-wild/blooming_rose/0.png

# (Optional) static 3D generation with DG
python dg.py --config configs/dg.yaml input=data/CONSISTENT4D_DATA/in-the-wild/blooming_rose/0.png
```

##### Step 2: dynamic generation
```bash
python main_4d.py --config configs/4d_c4d.yaml input=data/CONSISTENT4D_DATA/in-the-wild/blooming_rose

# (Optional) to load static 3D from DG, add `radius=2`
python main_4d.py --config configs/4d_c4d.yaml input=data/CONSISTENT4D_DATA/in-the-wild/blooming_rose radius=2
```
## Run demo locally
```bash
gradio gradio_app.py
```

## Load exported meshes in Blender
- Install the [Stop-motion-OBJ
](https://github.com/neverhood311/Stop-motion-OBJ) add-on
- File -> Import -> Mesh Sequence
- Go to `logs` directory, type in the file name (e.g., 'anya'), and tick `Material per Frame`.

https://github.com/jiawei-ren/dreamgaussian4d/assets/72253125/a558a475-e2db-4cdf-9bbf-e0e8d031e232


## Tips
 
- Black video after running `gen_vid.py`.
    - Make sure pytorch version is >=2.0 



## Acknowledgement

This work is built on many amazing research works and open-source projects, thanks a lot to all the authors for sharing!
* [4DGaussians](https://github.com/hustvl/4DGaussians)
* [DreamGaussian](https://github.com/dreamgaussian/dreamgaussian)
* [gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) and [diff-gaussian-rasterization](https://github.com/graphdeco-inria/diff-gaussian-rasterization)
* [threestudio](https://github.com/threestudio-project/threestudio)
* [nvdiffrast](https://github.com/NVlabs/nvdiffrast)

## Citation

```
@article{ren2023dreamgaussian4d,
  title={DreamGaussian4D: Generative 4D Gaussian Splatting},
  author={Ren, Jiawei and Pan, Liang and Tang, Jiaxiang and Zhang, Chi and Cao, Ang and Zeng, Gang and Liu, Ziwei},
  journal={arXiv preprint arXiv:2312.17142},
  year={2023}
}
```

