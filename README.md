
## In progress
- An experiment to encode geometry, material information for NVAS




## Installation
create the basic environment
```sh
conda env create --file environment.yml
conda activate gsir

pip install kornia
```

install some extensions
```sh
cd gs-ir && python setup.py develop && cd ..

cd submodules
git clone https://github.com/NVlabs/nvdiffrast
pip install ./nvdiffrast

pip install ./simple-knn
pip install ./diff-gaussian-rasterization # or cd ./diff-gaussian-rasterization && python setup.py develop && cd ../..  
```  

- python: 3.8.19
- CUDA: 11.8




## Running
 - Refer to [this site](https://caramel-process-7ad.notion.site/Code-a6edb3c9d3a94863b3808f023ac7d396?pvs=4) for some issues.



## Citation
```txt
@article{liang2023gs,
  title={Gs-ir: 3d gaussian splatting for inverse rendering},
  author={Liang, Zhihao and Zhang, Qi and Feng, Ying and Shan, Ying and Jia, Kui},
  journal={arXiv preprint arXiv:2311.16473},
  year={2023}
}
```
