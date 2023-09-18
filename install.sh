# pytorch
###
 # @Author: daniel
 # @Date: 2023-05-05 17:45:18
 # @LastEditTime: 2023-08-19 16:10:35
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /openset_anomaly_detection/installation/install.sh
 # have a nice day
### 




conda create -n pad python=3.8

conda activate pad


conda install pytorch -c pytorch -c https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/pytorch/linux-64/


conda install torchvision torchaudio cudatoolkit=11.6 -c pytorch -c conda-forge
# yaml
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple PyYAML
# Cython
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple Cython
# torch-scatter
conda install pytorch-scatter -c pyg
# nuScenes-devkit
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple nuscenes-devkit
# spconv
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple spconv-cu114
# numba
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple numba
# strictyaml
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple strictyaml





pip install scikit-image 
pip install astropy
pip install loguru
pip install wandb

pip install plyfile
pip install trimesh




python -c "import torch; print(torch.cuda.is_available())"



