import numpy as np
import torch
import argparse
from experiment_launcher import run_experiment
from experiment_launcher.launcher import add_launcher_base_args, get_experiment_default_params
import os
import time
import json
import dmc2gym
import datetime

import utils
from logger import Logger
from video import VideoRecorder

from dypre_sac import DypreSacAgent


def parse_args():
    parser = argparse.ArgumentParser()
    # environment
    parser.add_argument('--domain-name', type=str)
    parser.add_argument('--task-name', type=str)
    parser.add_argument('--action-repeat', type=int)
    parser.add_argument('--num-train-steps', type=int)
    parser.add_argument('--results-dir', type=str)  # modify

    # Hyperparameters
    parser.add_argument('--dypre-lr', type=float)
    parser.add_argument('--omega-dypre-loss', type=float)
    parser.add_argument('--time-step', type=int)
    parser.add_argument('--intrinsic-reward-scale', type=float)
    parser.add_argument('--use-external-reward', action='store_true')
    parser.add_argument('--fc-output-logits', type=float)
    parser.add_argument('--kl-use-target', type=float)

    # natural distractor
    parser.add_argument('--add-distractor', action='store_true')
    parser.add_argument('--img-source', type=str)
    parser.add_argument('--total-frames', type=int)
    # parser.add_argument('--resource-files', type=str)
    # parser.add_argument('--eval-resource-files', type=str)

    parser.add_argument('--pre-transform-image-size', type=int)
    parser.add_argument('--image-size', type=int)
    parser.add_argument('--frame-stack', type=int)
    # replay buffer
    parser.add_argument('--replay-buffer-capacity', type=int)
    # train
    parser.add_argument('--agent', type=str)
    parser.add_argument('--init-steps', type=int)

    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--hidden-dim', type=int)
    # eval
    parser.add_argument('--eval-freq', type=int)
    parser.add_argument('--num-eval-episodes', type=int)
    # critic
    parser.add_argument('--critic-lr', type=float)
    parser.add_argument('--critic-beta', type=float)
    parser.add_argument('--critic-tau', type=float)  # try 0.05 or 0.1
    parser.add_argument('--critic-target-update-freq',
                        type=int)  # try to change it to 1 and retain 0.01 above
    # actor
    parser.add_argument('--actor-lr', type=float)
    parser.add_argument('--actor-beta', type=float)
    parser.add_argument('--actor-log-std-min', type=float)
    parser.add_argument('--actor-log-std-max', type=float)
    parser.add_argument('--actor-update-freq', type=int)
    # encoder
    parser.add_argument('--encoder-type', type=str)
    parser.add_argument('--encoder-feature-dim', type=int)
    parser.add_argument('--encoder-lr', type=float)
    parser.add_argument('--encoder-tau', type=float)
    parser.add_argument('--num-layers', type=int)
    parser.add_argument('--num-filters', type=int)
    parser.add_argument('--dypre-latent-dim', type=int)
    # sac
    parser.add_argument('--discount', type=float)
    parser.add_argument('--init-temperature', type=float)
    parser.add_argument('--alpha-lr', type=float)
    parser.add_argument('--alpha-beta', type=float)
    # misc
    parser.add_argument('--seed', type=int)

    parser.add_argument('--save-tb', action='store_true')
    parser.add_argument('--save-buffer', action='store_true')
    parser.add_argument('--save-video', action='store_true')
    parser.add_argument('--save-model', action='store_true')
    parser.add_argument('--save-embedding', action='store_true')
    parser.add_argument('--detach-encoder', action='store_true')

    parser.add_argument('--log-interval', type=int)

    parser = add_launcher_base_args(parser)
    parser.set_defaults(**get_experiment_default_params(experiment))
    args = parser.parse_args()
    return vars(args)

def evaluate(env, agent, video, num_episodes, L, step, args, viz=False, device=None, embed_viz_dir=None):
    '''
    Evaluate the agent

    env:
    agent:
    video:
    num_episodes: the number of episodes per evaluation
    '''
    all_ep_rewards = []

    obses = []
    rewards_vis = []
    embeddings = []

    def run_eval_loop(sample_stochastically=True):
        start_time = time.time()
        prefix = 'stochastic_' if sample_stochastically else ''
        for i in range(num_episodes):
            obs = env.reset()
            video.init(enabled=(i == 0))
            done = False
            episode_reward = 0
            while not done:
                with utils.eval_mode(agent):
                    if sample_stochastically:
                        action = agent.sample_action(obs)
                    else:
                        action = agent.select_action(obs)

                obs, reward, done, _ = env.step(action)
                video.record(env)
                episode_reward += reward

                if viz:
                    obses.append(obs)
                    with torch.no_grad():
                        # Note that reward in [0, 1 * action_repeat]
                        rewards_vis.append(reward)
                        _, _, state, _ = agent.critic.encoder(torch.Tensor(obs).unsqueeze(0).to(device))
                        embeddings.append(state.cpu().detach().numpy())

            video.save('%d.mp4' % step)
            L.log('eval/' + prefix + 'episode_reward', episode_reward, step)
            all_ep_rewards.append(episode_reward)

        if viz:
            dataset = {'obs': obses, 'rewards': rewards_vis, 'embeddings': embeddings}
            torch.save(dataset, '%s/train_dataset_%s.pt' % (embed_viz_dir, step))

        L.log('eval/' + prefix + 'eval_time', time.time() - start_time, step)
        mean_ep_reward = np.mean(all_ep_rewards)
        best_ep_reward = np.max(all_ep_rewards)
        L.log('eval/' + prefix + 'mean_episode_reward', mean_ep_reward, step)
        L.log('eval/' + prefix + 'best_episode_reward', best_ep_reward, step)

    run_eval_loop(sample_stochastically=False)
    L.dump(step)


def make_agent(obs_shape, action_shape, args, device, action_repeat):
    if args.agent == 'dypre_sac':
        return DypreSacAgent(
            obs_shape=obs_shape,
            action_shape=action_shape,
            device=device,
            action_repeat=action_repeat,
            hidden_dim=args.hidden_dim,
            discount=args.discount,
            init_temperature=args.init_temperature,
            alpha_lr=args.alpha_lr,
            alpha_beta=args.alpha_beta,
            actor_lr=args.actor_lr,
            actor_beta=args.actor_beta,
            actor_log_std_min=args.actor_log_std_min,
            actor_log_std_max=args.actor_log_std_max,
            actor_update_freq=args.actor_update_freq,
            critic_lr=args.critic_lr,
            critic_beta=args.critic_beta,
            critic_tau=args.critic_tau,
            critic_target_update_freq=args.critic_target_update_freq,
            encoder_type=args.encoder_type,
            encoder_feature_dim=args.encoder_feature_dim,
            encoder_lr=args.encoder_lr,
            encoder_tau=args.encoder_tau,
            num_layers=args.num_layers,
            num_filters=args.num_filters,
            log_interval=args.log_interval,
            detach_encoder=args.detach_encoder,
            dypre_latent_dim=args.dypre_latent_dim,
            dypre_lr=args.dypre_lr,
            omega_dypre_loss=args.omega_dypre_loss,
            time_step=args.time_step,
            intrinsic_reward_scale=args.intrinsic_reward_scale,
            use_external_reward=args.use_external_reward,
            kl_use_target=args.kl_use_target,
            fc_output_logits=args.fc_output_logits
        )
    else:
        assert 'agent is not supported: %s' % args.agent


def experiment(
        domain_name: str = 'cartpole',
        task_name: str = 'swingup',
        pre_transform_image_size: int = 84,
        image_size: int = 84,
        action_repeat: int = 8,
        frame_stack: int = 3,
        replay_buffer_capacity: int = 100000,
        agent: str = 'dypre_sac',
        init_steps: int = 1000,
        num_train_steps: int = 63000,
        batch_size: int = 128,
        hidden_dim: int = 1024,
        eval_freq: int = 10000,
        num_eval_episodes: int = 10,
        critic_lr: float = 1e-3,
        critic_beta: float = 0.9,
        critic_tau: float = 0.01,
        critic_target_update_freq: int = 2,
        actor_lr: float = 1e-3,
        actor_beta: float = 0.9,
        actor_log_std_min: float = -10,
        actor_log_std_max: float = 2,
        actor_update_freq: int = 2,
        encoder_type: str = 'pixel',
        encoder_feature_dim: int = 50,
        encoder_lr: float = 1e-3,
        encoder_tau: float = 0.05,
        num_layers: int = 4,
        num_filters: int = 32,
        dypre_latent_dim: int = 128,
        discount: float = 0.99,
        init_temperature: float = 0.1,
        alpha_lr: float = 1e-4,
        alpha_beta: float = 0.5,
        save_tb: bool = True,
        save_buffer: bool = False,
        save_video: bool = False,
        save_model: bool = False,
        save_embedding: bool = False,
        detach_encoder: bool = False,
        dypre_lr: float = 1e-4,
        omega_dypre_loss: float = 1e-5,
        time_step: int = 2,
        intrinsic_reward_scale: float = 0.1,
        use_external_reward: bool = True,
        kl_use_target: bool = True,
        fc_output_logits: bool = True,
        add_distractor: bool = False,
        img_source: str = 'video',
        total_frames: int = 1000,
        log_interval: int = 100,
        seed: int = 0,
        results_dir: str = '/logs'
):

    resource_files = './'
    eval_resource_files = './'

    # check parameters
    args = utils.Namespace(
        domain_name=domain_name,
        task_name=task_name,
        pre_transform_image_size=pre_transform_image_size,
        image_size=image_size,
        action_repeat=action_repeat,
        frame_stack=frame_stack,
        replay_buffer_capacity=replay_buffer_capacity,
        agent=agent,
        init_steps=init_steps,
        num_train_steps=num_train_steps,
        batch_size=batch_size,
        hidden_dim=hidden_dim,
        eval_freq=eval_freq,
        num_eval_episodes=num_eval_episodes,
        critic_lr=critic_lr,
        critic_beta=critic_beta,
        critic_tau=critic_tau,
        critic_target_update_freq=critic_target_update_freq,
        actor_lr=actor_lr,
        actor_beta=actor_beta,
        actor_log_std_min=actor_log_std_min,
        actor_log_std_max=actor_log_std_max,
        actor_update_freq=actor_update_freq,
        encoder_type=encoder_type,
        encoder_feature_dim=encoder_feature_dim,
        encoder_lr=encoder_lr,
        encoder_tau=encoder_tau,
        num_layers=num_layers,
        num_filters=num_filters,
        dypre_latent_dim=dypre_latent_dim,
        discount=discount,
        init_temperature=init_temperature,
        alpha_lr=alpha_lr,
        alpha_beta=alpha_beta,
        save_tb=save_tb,
        save_buffer=save_buffer,
        save_video=save_video,
        save_model=save_model,
        save_embedding=save_embedding,
        detach_encoder=detach_encoder,
        dypre_lr=dypre_lr,
        omega_dypre_loss=omega_dypre_loss,
        time_step=time_step,
        intrinsic_reward_scale=intrinsic_reward_scale,
        use_external_reward=use_external_reward,
        kl_use_target=kl_use_target,
        fc_output_logits=fc_output_logits,
        resource_files=resource_files,
        eval_resource_files=eval_resource_files,
        add_distractor=add_distractor,
        img_source=img_source,
        total_frames=total_frames,
        log_interval=log_interval,
        seed=seed,
        results_dir=results_dir
    )

    os.makedirs(results_dir, exist_ok=True)
    utils.set_seed_everywhere(seed)


    env = dmc2gym.make(
        domain_name=args.domain_name,
        task_name=args.task_name,
        seed=args.seed,
        visualize_reward=False,
        from_pixels=(args.encoder_type == 'pixel'),
        height=args.pre_transform_image_size,
        width=args.pre_transform_image_size,
        # Setting frame_skip argument lets to perform action repeat
        frame_skip=args.action_repeat
    )

    env.seed(args.seed)

    # stack several consecutive frames together
    if args.encoder_type == 'pixel' and not args.add_distractor:
        env = utils.FrameStack(env, k=args.frame_stack)

    # make directory
    ts = datetime.datetime.fromtimestamp(time.time()).strftime("%m-%d-%H-%M")
    env_name = args.domain_name + '-' + args.task_name
    exp_name = env_name + '-s' + str(args.seed)
    args.results_dir = args.results_dir + '/' + 'dypre_' + args.domain_name + '/' + exp_name
    print(args.results_dir)

    utils.make_dir(args.results_dir)
    video_dir = utils.make_dir(os.path.join(args.results_dir, 'video'))
    model_dir = utils.make_dir(os.path.join(args.results_dir, 'model'))
    buffer_dir = utils.make_dir(os.path.join(args.results_dir, 'buffer'))
    embedding_dir = utils.make_dir(os.path.join(args.results_dir, 'embedding'))

    video = VideoRecorder(video_dir if args.save_video else None)

    # open the file args.json in work_dir and write args into args.json
    with open(os.path.join(args.results_dir, 'args.json'), 'w') as f:
        json.dump(vars(args), f, sort_keys=True, indent=4)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    action_shape = env.action_space.shape

    if args.encoder_type == 'pixel':
        obs_shape = (3 * args.frame_stack, args.image_size, args.image_size)
        pre_aug_obs_shape = (3 * args.frame_stack, args.pre_transform_image_size, args.pre_transform_image_size)
    else:
        obs_shape = env.observation_space.shape
        pre_aug_obs_shape = obs_shape

    replay_buffer = utils.ReplayBuffer(
        obs_shape=pre_aug_obs_shape,
        action_shape=action_shape,
        capacity=args.replay_buffer_capacity,
        batch_size=args.batch_size,
        device=device,
        image_size=args.image_size,
    )

    agent = make_agent(
        obs_shape=obs_shape,
        action_shape=action_shape,
        args=args,
        device=device,
        action_repeat=args.action_repeat
    )

    # tensorboard visualization
    L = Logger(args.results_dir, use_tb=args.save_tb)

    episode, episode_reward, done = 0, 0, True
    start_time = time.time()

    for step in range(args.num_train_steps):
        # evaluate agent periodically

        if step % args.eval_freq == 0:
            L.log('eval/episode', episode, step)
            evaluate(env, agent, video, args.num_eval_episodes, L, step, args, viz=args.save_embedding, embed_viz_dir=embedding_dir, device=device)
            if args.save_model:
                agent.save_dypre(model_dir, step)
            if args.save_buffer:
                replay_buffer.save(buffer_dir)

        if done:
            if step > 0:
                if step % args.log_interval == 0:
                    L.log('train/duration', time.time() - start_time, step)
                    # log data into train.log
                    L.dump(step)
                start_time = time.time()
            if step % args.log_interval == 0:
                L.log('train/episode_reward', episode_reward, step)

            # expand one frame image into k frames by copying when resetting env
            obs = env.reset()
            done = False
            episode_reward = 0
            episode_step = 0
            episode += 1
            if step % args.log_interval == 0:
                L.log('train/episode', episode, step)

        # sample action for data collection
        if step < args.init_steps:
            action = env.action_space.sample()
        else:
            with utils.eval_mode(agent):
                action = agent.sample_action(obs)

        # run training update
        if step >= args.init_steps:
            num_updates = 1
            for _ in range(num_updates):
                agent.update(replay_buffer, L, step)

        next_obs, reward, done, _ = env.step(action)
        if done and episode_step + 1 < env._max_episode_steps:
            print("episode finished")


        # limit infinit bootstrap
        done_bool = 1 if episode_step + 1 == env._max_episode_steps else float(
            done
        )
        episode_reward += reward
        replay_buffer.add(obs, action, reward, next_obs, done_bool)

        obs = next_obs
        episode_step += 1


if __name__ == '__main__':
    torch.multiprocessing.set_start_method('spawn')

    run_experiment(experiment)
