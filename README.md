# Dynamic Bottleneck with a Predictable Prior for Image-based Deep Reinforcement Learning (DY2P)  

This repository is the official implementation of DY2P. Our implementation is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae).   

## Requirements  
Required dependencies of this repo can be installed by running:  
```sh
conda env create -f environment.yml  
```
Then you can activate the environment by running:  
```sh
source activate py3.6  
```
## Instructions
To train a DY2P agent on the ```cartpole swingup ``` task from images, you can run:
```
python train.py \
            --domain_name cartpole  \
            --task_name swingup  \
            --action_repeat 8 \
            --save_tb \
            --seed 0 \
            --eval_freq 1250 \
            --batch_size 512 \
            --pre_transform_image_size 84 \
            --image_size 84 \
            --cody_lr 1e-4 \
            --results_dir ./logs \
            --time_step 2 \
            --omega_cody_loss 0.01 \
            --fc_output_logits True \
            --kl_use_target True \
```
or you can run the script for all six tasks:
```
bash train.sh
```

