# Integrating Contrastive Learning with Dynamic Modelsfor Reinforcement Learning from Images   

This repository is the official pytorch implementation of CoDy for the DeepMind control experiments. Our implementation of SAC is based on [SAC+AE](https://github.com/denisyarats/pytorch_sac_ae).   

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
To train a CoDy+SAC agent on the ```cartpole swingup ``` task from images, you can run:
```sh
python train.py \
    --domain_name cartpole \
    --task_name swingup \
    --action_repeat 8 \
    --save_tb \
    --frame_stack 3 \
    --seed -1 \
    --eval_freq 10000 \
    --batch_size 256 \
    --num_train_steps 1000000 
```

## Todolist:
1. Test sample efficiency on different tasks  
2. Training stability
- [x] remove kl term  
- [x] remove ln in fc1 and fc1 target   
- [x] remove reward
- [x] remove action_encoder  
- [x] add projection  
- [ ] build a prediction head with more layers
- [ ] use online encoding for kl term  
- [ ] online fc   
- [ ] online encoder + fc
- [ ] tie the encoder with the target encoder
- [x] version problem, failed
- [ ] change small hypers
- [x] use protoRL-style encoder, tested: poor performance without action norm problem, which might be caused by using the same encoder for current observation and next observation

3. Training stability result  
4.1. remove kl loss and ln in fc, failed
4.2. remove action encoder and add projection, failed
4.3. test on the old env and remove kl, failed
4.4. remove kl loss and reward and ln, (on 2070 403), bingo! 
4.5. remove kl loss and reward and add proj, the result is little worse than the one without the proj.
4.6. normalize_reward, no ln, (on 2070 424), well done! The performance is not good but the training is stable. 
4.7. normalize_reward, remove ln, (on 2060), failed
Conclusion: bug is caused by the not normalized rewards which is computed by accumulating the reward over multiple repeated steps.

4. Hyperparameter optimization  
- [x] remove kl and reward and ln, good, baseline
- [x] remove ln  
the performance is really poor and kl value is really large (1e4). Besides, the training is not stable.
- [ ] normal (with kl, normalized reward and ln), 2070 2
The ln layer in fc model can significantly decrease the kl value (1e2).
- [ ] build a prediction head with more layers, 
- [ ] use online encoding for kl term, 2070 1,
- [ ] norm reward to [-1, 1]
- [ ] test omega: [0.0, 0.1, 0.01, 0.001, 0.0001, 1e-5], 0.001 on 2070 1, the rest on 2070 2 for cody_lr=1e-4, encoder_lr=1e-3
0.1, curshed with large kl 424. The potential reason might be the large gradient flow.
0.0001, crushed with small kl 4, 
0.001, poor performance with seed 3, while the mi is keeping at around 1.2.


test omega: [0.1, 0.0001, 1e-3], for cody_lr=1e-4, encoder_lr=1e-4    
0.001, poor performance    
0.1, crushed   

online target for kl and remove ln  

- [ ] omega: [0.1, 0.01, 0.001, 0.0001, 1e-5] 2070 lei
- [ ] omega: [0.1, 0.01, 0.001, 0.0001, 1e-5], use on_line encoder for kl, 2070 
- [ ] omega: [0.1, 0.01, 0.001, 0.0001, 1e-5], with nonlinear projection, 2060 
- [ ] omega: [0.001]

Result: 
1. omega=0.001 is the best.   
2. The nonlinear proj can slightly improve the performance.  
3. Using target encoder for kl is better than using the online encoder.
4. Using Nonlinear projection helps the performance.
5. kl is increasing during training and mi is remaining a large value 1.4 for cheetah run task.
6. cheetah run 512 is better than the one with 256.


5. Why performance varies across seeds?

6. ablation

- [ ] remove kl, 2070
- [ ] using unit Gaussion, 2070
- [ ] remove reward
- [ ] remove projection