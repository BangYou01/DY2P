import torch
import numpy as np
import torch.nn as nn
import gym
import os
from collections import deque
import random
from torch.utils.data import Dataset
import time
from skimage.util.shape import view_as_windows
import kornia


class eval_mode(object):
    def __init__(self, *models):
        self.models = models

    def __enter__(self):
        self.prev_states = []
        for model in self.models:
            self.prev_states.append(model.training)
            model.train(False)

    def __exit__(self, *args):
        for model, state in zip(self.models, self.prev_states):
            model.train(state)
        return False


def soft_update_params(net, target_net, tau):
    for param, target_param in zip(net.parameters(), target_net.parameters()):
        target_param.data = param.data * tau + (1 - tau) * target_param.data


def set_seed_everywhere(seed):
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


def module_hash(module):
    result = 0
    for tensor in module.state_dict().values():
        result += tensor.sum().item()
    return result


def make_dir(dir_path):
    try:
        os.makedirs(dir_path)
    except OSError:
        pass
    return dir_path


def preprocess_obs(obs, bits=5):
    """Preprocessing image, see https://arxiv.org/abs/1807.03039."""
    bins = 2 ** bits
    assert obs.dtype == torch.float32
    if bits < 8:
        obs = torch.floor(obs / 2 ** (8 - bits))
    obs = obs / bins
    obs = obs + torch.rand_like(obs) / bins
    obs = obs - 0.5
    return obs


class ReplayBuffer(Dataset):
    """Buffer to store environment transitions."""

    def __init__(self, obs_shape, action_shape, capacity, batch_size, device, image_size=84, transform=None):
        self.capacity = capacity
        self.batch_size = batch_size
        self.device = device
        self.image_size = image_size
        self.transform = transform
        # the proprioceptive obs is stored as float32, pixels obs as uint8
        obs_dtype = np.float32 if len(obs_shape) == 1 else np.uint8

        self.obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.next_obses = np.empty((capacity, *obs_shape), dtype=obs_dtype)
        self.actions = np.empty((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.empty((capacity, 1), dtype=np.float32)
        self.not_dones = np.empty((capacity, 1), dtype=np.float32)

        self.idx = 0
        self.last_save = 0
        self.full = False

        image_pad = 4
        self.aug_trans = nn.Sequential(
            nn.ReplicationPad2d(image_pad),
            kornia.augmentation.RandomCrop((obs_shape[-1], obs_shape[-1])))

    def add(self, obs, action, reward, next_obs, done):

        np.copyto(self.obses[self.idx], obs)
        np.copyto(self.actions[self.idx], action)
        np.copyto(self.rewards[self.idx], reward)
        np.copyto(self.next_obses[self.idx], next_obs)
        np.copyto(self.not_dones[self.idx], not done)

        self.idx = (self.idx + 1) % self.capacity
        self.full = self.full or self.idx == 0

    def sample_proprio(self):
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]

        obses = torch.as_tensor(obses, device=self.device).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)
        return obses, actions, rewards, next_obses, not_dones

    def sample_cpc(self):
        """
        sample a minibatch data and acquire instance discrimination by randomly cropping
        """
        idxs = np.random.randint(
            0, self.capacity if self.full else self.idx, size=self.batch_size
        )

        obses = self.obses[idxs]
        next_obses = self.next_obses[idxs]
        pos = obses.copy()

        obses = torch.as_tensor(obses, device=self.device).float()
        next_obses = torch.as_tensor(
            next_obses, device=self.device
        ).float()
        actions = torch.as_tensor(self.actions[idxs], device=self.device)
        rewards = torch.as_tensor(self.rewards[idxs], device=self.device)
        not_dones = torch.as_tensor(self.not_dones[idxs], device=self.device)

        pos = torch.as_tensor(pos, device=self.device).float()

        obses = self.aug_trans(obses)
        next_obses = self.aug_trans(next_obses)
        pos = self.aug_trans(pos)

        cpc_kwargs = dict(obs_anchor=obses, obs_pos=pos,
                          time_anchor=None, time_pos=None)

        return obses, actions, rewards, next_obses, not_dones, cpc_kwargs

    def sample_consecutive(self, t):
        """
        sample a consecutive minibatch data with time length tand acquire instance discrimination by randomly cropping
        """
        valid_idxs = np.array([], dtype=int)

        while not valid_idxs.size == self.batch_size:
            idxs = np.random.randint(
                0, self.capacity - t if self.full else self.idx - t, size=self.batch_size
            )

            invalid_index = np.array([], dtype=int)

            # allow     [1,1,..., 0]
            # time_step:[1,2,..., t]
            for i in range(t - 1):
                idxs_tem = idxs + i
                not_dones = self.not_dones[idxs_tem]
                invalid_index_tem = np.nonzero(not_dones == 0.0)[0]
                invalid_index = np.concatenate([invalid_index, invalid_index_tem])

            if invalid_index.size == 0:
                valid_idxs = idxs
            else:
                idxs = np.delete(idxs, invalid_index)
                required_size = self.batch_size - valid_idxs.size
                if required_size > idxs.size:
                    valid_idxs = np.concatenate([valid_idxs, idxs])
                else:
                    valid_idxs = np.concatenate([valid_idxs, idxs[:required_size]])

        sampled_obs = []
        sampled_actions = []
        sampled_next_obss = []
        sampled_reward = []
        sampled_not_done = []

        not_dones = torch.as_tensor(self.not_dones[valid_idxs], device=self.device)
        sampled_not_done.append(not_dones)
        sampled_not_done.append(not_dones)

        for i in range(t):
            obses = self.obses[valid_idxs]
            next_obses = self.next_obses[valid_idxs]

            obses = torch.as_tensor(obses, device=self.device).float()
            next_obses = torch.as_tensor(
                next_obses, device=self.device
            ).float()
            actions = torch.as_tensor(self.actions[valid_idxs], device=self.device)
            rewards = torch.as_tensor(self.rewards[valid_idxs], device=self.device)

            obses = self.aug_trans(obses)
            next_obses = self.aug_trans(next_obses)

            sampled_obs.append(obses)
            sampled_actions.append(actions)
            sampled_reward.append(rewards)
            sampled_next_obss.append(next_obses)

            valid_idxs += 1

        return sampled_obs, sampled_actions, sampled_reward, sampled_next_obss, sampled_not_done

    def save(self, save_dir):
        if self.idx == self.last_save:
            return
        path = os.path.join(save_dir, '%d_%d.pt' % (self.last_save, self.idx))
        payload = [
            self.obses[self.last_save:self.idx],
            self.next_obses[self.last_save:self.idx],
            self.actions[self.last_save:self.idx],
            self.rewards[self.last_save:self.idx],
            self.not_dones[self.last_save:self.idx]
        ]
        self.last_save = self.idx
        torch.save(payload, path)

    def load(self, save_dir):
        chunks = os.listdir(save_dir)
        chucks = sorted(chunks, key=lambda x: int(x.split('_')[0]))
        for chunk in chucks:
            start, end = [int(x) for x in chunk.split('.')[0].split('_')]
            path = os.path.join(save_dir, chunk)
            payload = torch.load(path)
            assert self.idx == start
            self.obses[start:end] = payload[0]
            self.next_obses[start:end] = payload[1]
            self.actions[start:end] = payload[2]
            self.rewards[start:end] = payload[3]
            self.not_dones[start:end] = payload[4]
            self.idx = end

    def __getitem__(self, idx):
        idx = np.random.randint(
            0, self.capacity if self.full else self.idx, size=1
        )
        idx = idx[0]
        obs = self.obses[idx]
        action = self.actions[idx]
        reward = self.rewards[idx]
        next_obs = self.next_obses[idx]
        not_done = self.not_dones[idx]

        if self.transform:
            obs = self.transform(obs)
            next_obs = self.transform(next_obs)

        return obs, action, reward, next_obs, not_done

    def __len__(self):
        return self.capacity


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self._k = k
        self._frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(
            low=0,
            high=1,
            shape=((shp[0] * k,) + shp[1:]),
            dtype=env.observation_space.dtype
        )
        self._max_episode_steps = env._max_episode_steps

    def reset(self):
        obs = self.env.reset()
        for _ in range(self._k):
            self._frames.append(obs)
        return self._get_obs()

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self._frames.append(obs)
        return self._get_obs(), reward, done, info

    def _get_obs(self):
        assert len(self._frames) == self._k
        return np.concatenate(list(self._frames), axis=0)


def random_crop(imgs, output_size):
    """
    Vectorized way to do random crop using sliding windows
    and picking out random ones

    args:
        imgs, batch images with shape (B,C,H,W)
    """
    # batch size
    n = imgs.shape[0]
    img_size = imgs.shape[-1]
    crop_max = img_size - output_size
    imgs = np.transpose(imgs, (0, 2, 3, 1))
    w1 = np.random.randint(0, crop_max, n)
    h1 = np.random.randint(0, crop_max, n)
    # creates all sliding windows combinations of size (output_size)
    windows = view_as_windows(
        imgs, (1, output_size, output_size, 1))[..., 0, :, :, 0]
    # selects a random window for each batch element
    cropped_imgs = windows[np.arange(n), w1, h1]
    return cropped_imgs


def center_crop_image(image, output_size):
    h, w = image.shape[1:]
    new_h, new_w = output_size, output_size

    top = (h - new_h) // 2
    left = (w - new_w) // 2

    image = image[:, top:top + new_h, left:left + new_w]
    return image


class Namespace:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def para_sum(net):
    para_dict = {}
    # grad_conv1_list = []
    for name, param in net.named_parameters():
        if name in para_dict:
            para_dict[name].append(param.data.sum())

        else:
            para_dict[name] = []
            para_dict[name].append(param.data.sum())

    return para_dict


def image_show(stack_obs):
    for i in range(3):
        obs = stack_obs[i * 3:i * 3 + 3, :, :]
        obs = obs.transpose(1, 2, 0)
        from PIL import Image
        image = Image.fromarray(obs)
        image.show()

class Normalize(nn.Module):
    def __init__(self, mean, std, action_repeat):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean)
        self.std = torch.tensor(std)
        self.range = action_repeat

    def forward(self, input):
        x = input / self.range
        x = x - self.mean
        x = x / self.std
        return x
