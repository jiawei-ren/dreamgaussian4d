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
  <a href="https://jiawei-ren.github.io/projects/dreamgaussian4d/" target='_blank'>[Project Page]</a> •
  <a href="https://arxiv.org/abs/2312.17142" target='_blank'>[Paper]</a>

</h4>

</div>

## Install
```bash
# install customized diffusers
pip install ./diffusers

pip install -r requirements.txt

# a modified gaussian splatting (+ depth, alpha rendering)
git clone --recursive https://github.com/ashawkey/diff-gaussian-rasterization
pip install ./diff-gaussian-rasterization

# simple-knn
pip install ./simple-knn

# nvdiffrast
pip install git+https://github.com/NVlabs/nvdiffrast/

# kiuikit
pip install git+https://github.com/ashawkey/kiuikit

```

Tested on:
*  torch 2.1 & CUDA 11.8 on an 80GB A100.

## Usage
```bash
# generate driving video
python gen_vid.py --name anya_rgba --seed 42 --bg white

# Stage I: train 500 iters (~2min) and export ckpt & coarse_mesh to logs
python main.py --config configs/image.yaml input=data/anya_rgba.png save_path=anya

# Stage II: temporal optimization stage
python main_4d.py --config configs/4d.yaml input=data/anya_rgba.png save_path=anya

# Stage III: texture optimization (optional, it requires large GPU memory and we are optimzing it)
python main2_4d.py --config configs/4d_svd.yaml input=data/anya_rgba.png save_path=anya

# to turn on viser GUI, add `gui=True`, e.g.:
python main.py --config configs/image.yaml input=data/anya_rgba.png save_path=anya gui=True
```
Meshes will be automatically exported to `logs` in Stage II. Visulizations will be saved to `vis_data`.

## Load exported meshes in Blender
- Install the [Stop-motion-OBJ
](https://github.com/neverhood311/Stop-motion-OBJ) add-on
- File -> Import -> Mesh Sequence
- Go to `logs` directory, type in the file name (e.g., 'anya'), and tick `Material per Frame`.
<img width="336" alt="Screenshot 2023-12-28 at 7 08 58 PM" src="https://github.com/jiawei-ren/dreamgaussian4d/assets/72253125/9c32436b-bbf9-432b-9bcb-3d3bcb3c1866">


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

