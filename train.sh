domain=(cartpole ball_in_cup reacher finger walker cheetah)
task=(swingup catch easy spin walk run)
ar=(8 4 4 2 2 4)
ns=(63000 125500 125500 250100 250100 250100)
ef=(1250 2500 2500 5000 5000 2500)

for i in ${!domain[*]}; do
    for s in 0 1 2 3 4; do
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
            --dypre_lr 1e-4 \
            --results_dir ./logs \
            --time_step 2 \
            --omega_dypre_loss 0.01 \
            --fc_output_logits True \
            --kl_use_target True \
            --num_train_steps ${ns[$i]}
    done
done
