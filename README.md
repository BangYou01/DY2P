# DY2P  

This repository is the official pytorch implementation of DY2P for the DeepMind control experiments. Our implementation is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae).   

## Requirements  
It is simple to install required dependencies by running:  
```sh
conda env create -f environment.yml  
```
Then you can activate the environment by running:  
```sh
source activate py3.6  
```
## Instructions
To train a DY2P agent on the ```cartpole swingup ``` task from images, you can run:
```sh
python train.py \
            --domain_name ${domain[$i]}  \
            --task_name ${task[$i]}  \
            --action_repeat ${ar[$i]} \
            --save_tb \
            --seed $s \
            --eval_freq ${ef[$i]} \
            --batch_size 512 \
            --pre_transform_image_size 84 \
            --image_size 84 \
            --cody_lr 1e-4 \
            --results_dir ./logs \
            --time_step 2 \
            --omega_cody_loss 0.01 \
            --fc_output_logits True \
            --kl_use_target True \
            --num_train_steps ${ns[$i]}
```
or you can run the script for all six tasks:
```sh train.sh
```

