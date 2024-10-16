# Dypre

This repository is the official implementation of Dypre. Our implementation is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae).   

## Requirements  
Required dependencies of this repo can be installed by running:  
```sh
conda env create --name dypre python==3.8  
pip install -r requirements.txt  
pip install git+https://github.com/denisyarats/dmc2gym.git  
conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch  
conda install tensorboard  
```
Then you can activate the environment by running:  
```sh
source activate py3.6  
```
## Instructions
To train a Dypre agent on the ```cartpole swingup ``` task with images,  run the command:
```
python train.py \
            --domain_name cartpole  \
            --task_name swingup  \
            --action_repeat 8 \
            --save_tb \
            --seed 0 \
            --eval_freq 1250 \
            --batch_size 512 \
            --results_dir ./logs \
            --time_step 2 \
            --omega_dypre_loss 0.01 \
            --fc_output_logits True \
            --kl_use_target True \
```
To evaluate Dypre on all 6 control tasks, run the train.sh scriptor:
```
bash train.sh
```

