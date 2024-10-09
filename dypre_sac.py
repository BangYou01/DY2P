import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque

import utils
from encoder import make_encoder, Transition_model

LOG_FREQ = 10000


def gaussian_logprob(noise, log_std):
    """Compute Gaussian log probability."""
    residual = (-0.5 * noise.pow(2) - log_std).sum(-1, keepdim=True)
    return residual - 0.5 * np.log(2 * np.pi) * noise.size(-1)


def squash(mu, pi, log_pi):
    """Apply squashing function.
    See appendix C from https://arxiv.org/pdf/1812.05905.pdf.
    """
    mu = torch.tanh(mu)
    if pi is not None:
        pi = torch.tanh(pi)
    if log_pi is not None:
        log_pi -= torch.log(F.relu(1 - pi.pow(2)) + 1e-6).sum(-1, keepdim=True)
    return mu, pi, log_pi


def weight_init(m):
    """Custom weight init for Conv2D and Linear layers."""
    if isinstance(m, nn.Linear):
        nn.init.orthogonal_(m.weight.data)
        m.bias.data.fill_(0.0)
    elif isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        # delta-orthogonal init from https://arxiv.org/pdf/1806.05393.pdf
        assert m.weight.size(2) == m.weight.size(3)
        m.weight.data.fill_(0.0)
        m.bias.data.fill_(0.0)
        mid = m.weight.size(2) // 2
        gain = nn.init.calculate_gain('relu')
        nn.init.orthogonal_(m.weight.data[:, :, mid, mid], gain)


class Actor(nn.Module):
    """MLP actor network."""

    def __init__(
            self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, log_std_min, log_std_max, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.feature_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 2 * action_shape[0])
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(
            self, obs, compute_pi=True, compute_log_pi=True, detach_encoder=False, encoder_use_mean=False
    ):
        obs = self.encoder(obs, detach=detach_encoder)

        mu, log_std = self.trunk(obs).chunk(2, dim=-1)

        log_std = torch.tanh(log_std)
        log_std = self.log_std_min + 0.5 * (
                self.log_std_max - self.log_std_min
        ) * (log_std + 1)

        self.outputs['mu'] = mu
        self.outputs['std'] = log_std.exp()

        if compute_pi:
            std = log_std.exp()
            noise = torch.randn_like(mu)
            pi = mu + noise * std
        else:
            pi = None
            entropy = None

        if compute_log_pi:
            log_pi = gaussian_logprob(noise, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = squash(mu, pi, log_pi)

        return mu, pi, log_pi, log_std

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_actor/%s_hist' % k, v, step)

        L.log_param('train_actor/fc1', self.trunk[0], step)
        L.log_param('train_actor/fc2', self.trunk[2], step)
        L.log_param('train_actor/fc3', self.trunk[4], step)


class QFunction(nn.Module):
    """MLP for q-function."""

    def __init__(self, obs_dim, action_dim, hidden_dim):
        super().__init__()

        self.trunk = nn.Sequential(
            nn.Linear(obs_dim + action_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, obs, action):
        assert obs.size(0) == action.size(0)

        obs_action = torch.cat([obs, action], dim=1)
        return self.trunk(obs_action)


class Critic(nn.Module):
    """Critic network, employes two q-functions."""

    def __init__(
            self, obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
    ):
        super().__init__()

        self.encoder = make_encoder(
            encoder_type, obs_shape, encoder_feature_dim, num_layers,
            num_filters, output_logits=True
        )

        self.Q1 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )
        self.Q2 = QFunction(
            self.encoder.feature_dim, action_shape[0], hidden_dim
        )

        self.outputs = dict()
        self.apply(weight_init)

    def forward(self, obs, action, detach_encoder=False):

        obs = self.encoder(obs, detach=detach_encoder)

        q1 = self.Q1(obs, action)
        q2 = self.Q2(obs, action)

        self.outputs['q1'] = q1
        self.outputs['q2'] = q2

        return q1, q2

    def log(self, L, step, log_freq=LOG_FREQ):
        if step % log_freq != 0:
            return

        self.encoder.log(L, step, log_freq)

        for k, v in self.outputs.items():
            L.log_histogram('train_critic/%s_hist' % k, v, step)

        for i in range(3):
            L.log_param('train_critic/q1_fc%d' % i, self.Q1.trunk[i * 2], step)
            L.log_param('train_critic/q2_fc%d' % i, self.Q2.trunk[i * 2], step)


class Dypre(nn.Module):
    """
    Auxiliary task
    """

    def __init__(self, z_dim, critic, critic_target, action_shape,  output_type="continuous", fc_output_logits=True, kl_use_target=True):
        super().__init__()


        # Build online encoders
        self.rgb_img_encoder = critic.encoder
        self.rgb_img_encoder_target = critic_target.encoder

        self.kl_use_target = kl_use_target


        # MLP
        self.fc1 = Transition_model(action_shape[0], z_dim, output_feature_num=z_dim + 1, num_layers=2,
                                    output_logits=fc_output_logits)
        self.fc2 = Transition_model(action_shape[0], z_dim, output_feature_num=z_dim + 1, num_layers=2,
                                           output_logits=fc_output_logits)

        self.projection = nn.Sequential(nn.Linear(z_dim + 1, 1024), nn.ReLU(), nn.Linear(1024, z_dim + 1))
        self.predictor = nn.Sequential(nn.Linear(z_dim + 1, 1024), nn.ReLU(), nn.Linear(1024, z_dim))
        self.predictor_target = nn.Sequential(nn.Linear(z_dim + 1, 1024), nn.ReLU(), nn.Linear(1024, z_dim))

        # Initialize target encoders
        for param_rgb_encoder, param_rgb_encoder_target in zip(self.rgb_img_encoder.parameters(),
                                                               self.rgb_img_encoder_target.parameters()):
            param_rgb_encoder_target.data.copy_(param_rgb_encoder.data)
            param_rgb_encoder_target.requires_grad = False

        for param_predictor, param_predictor_target in zip(self.predictor.parameters(),
                                                           self.predictor_target.parameters()):
            param_predictor_target.data.copy_(param_predictor.data)
            param_predictor_target.requires_grad = False

        self.W = nn.Parameter(torch.rand(z_dim, z_dim))

        self.W.requires_grad == True

        self.output_type = output_type

        self.apply(weight_init)

    def encode(self, obs, action, next_obs, action_condition=False):

        # current_step samples
        z = self.rgb_img_encoder(obs)

        # next_step samples
        with torch.no_grad():
            z_next = self.rgb_img_encoder_target(next_obs)

        #action_embed = self.action_encoder(action)
        [mu, logstd, z_cat] = self.fc1(z, action, condition=action_condition)

        return z_cat, z_next, [mu, logstd]

    def reparameterize(self, mu, logstd, output_mean=False):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)

        if output_mean:
            output = mu
        else:
            output = mu + eps * std
        return output

    def kl_divergence(self, mu_p, logstd_p, obs_old, action_old):
        """
        compute the kl divergence between p(z/s,a) and q(z)
        """

        if self.kl_use_target:
            z_old_target = self.rgb_img_encoder_target(obs_old)
            [mu_q, logstd_q, _] = self.fc2(z_old_target.detach(), action_old, condition=True)

        kl = logstd_q - logstd_p + (torch.exp(logstd_p) ** 2 + (mu_p - mu_q) ** 2) / (
                    2 * torch.exp(logstd_q) ** 2) - 0.5
        kl = kl.sum(dim=-1).mean()
        return kl

    def compute_logits(self, z_a, z_pos, reward):
        z_a = self.projection(z_a)
        z_a = self.predictor(z_a)
        z_pos_rewards = torch.cat([z_pos, reward], dim=1)
        with torch.no_grad():
            z_pos = self.predictor_target(z_pos_rewards)

        Wz = torch.matmul(self.W, z_pos.T)  # (z_dim,B)
        logits = torch.matmul(z_a, Wz)  # (B,B)
        logits = logits - torch.max(logits, 1)[0][:, None]

        return logits


class DypreSacAgent(object):
    """Representation learning with SAC."""
    def __init__(
            self,
            obs_shape,
            action_shape,
            device,
            action_repeat,
            hidden_dim=256,
            discount=0.99,
            init_temperature=0.01,
            alpha_lr=1e-3,
            alpha_beta=0.9,
            actor_lr=1e-3,
            actor_beta=0.9,
            actor_log_std_min=-10,
            actor_log_std_max=2,
            actor_update_freq=2,
            critic_lr=1e-3,
            critic_beta=0.9,
            critic_tau=0.005,
            critic_target_update_freq=2,
            encoder_type='pixel',
            encoder_feature_dim=50,
            encoder_lr=1e-3,
            encoder_tau=0.005,
            num_layers=4,
            num_filters=32,
            cpc_update_freq=1,
            log_interval=100,
            detach_encoder=False,
            dypre_latent_dim=128,
            dypre_lr=1e-7,
            omega_dypre_loss=0.1,
            beta_dypre=1,
            time_step=3,
            intrinsic_reward_scale=0.1,
            use_external_reward=True,
            kl_use_target=True,
            fc_output_logits=True,
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.cpc_update_freq = cpc_update_freq
        self.log_interval = log_interval
        self.image_size = obs_shape[-1]
        self.dypre_latent_dim = dypre_latent_dim
        self.detach_encoder = detach_encoder
        self.encoder_type = encoder_type
        self.omega_dypre_loss = omega_dypre_loss
        self.beta_dypre = beta_dypre
        self.time_step = time_step
        self.intrinsic_reward_scale = intrinsic_reward_scale
        self.use_external_reward = use_external_reward
        self.kl_use_target = kl_use_target
        self.fc_output_logits = fc_output_logits
        self.action_repeat = action_repeat

        self.transition_model_target_update_freq = 2
        self.transition_model_tau = 0.005
        ###
        self._action_frames = deque([], maxlen=3)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())
        for param_encoder, param_encoder_target in zip(self.critic.encoder.parameters(),
                                                       self.critic_target.encoder.parameters()):
            param_encoder_target.data.copy_(param_encoder.data)  # initialize
            param_encoder_target.requires_grad = False

        # tie encoders between actor and critic, and dypre and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        if self.encoder_type == 'pixel':
            self.dypre = Dypre(encoder_feature_dim, self.critic, self.critic_target, action_shape,
                             output_type='continuous', fc_output_logits=fc_output_logits, kl_use_target=kl_use_target).to(self.device)

            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

            self.cpc_optimizer = torch.optim.Adam(
                self.dypre.parameters(), lr=dypre_lr
            )
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.cross_entropy_loss_no_reduction = nn.CrossEntropyLoss(reduction='none')
        self.mse_loss = nn.MSELoss()

        self.train()
        self.critic_target.train()

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.encoder_type == 'pixel':
            self.dypre.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False, encoder_use_mean=True
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        if obs.shape[-1] != self.image_size:
            obs = utils.center_crop_image(obs, self.image_size)

        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=self.detach_encoder)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)

        if step % self.log_interval == 0:
            L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        if step % self.log_interval == 0:
            L.log('train_actor/loss', actor_loss, step)
            L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * \
                  (1.0 + np.log(2 * np.pi)) + log_std.sum(dim=-1)
        if step % self.log_interval == 0:
            L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        if step % self.log_interval == 0:
            L.log('train_alpha/loss', alpha_loss, step)
            L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()


    def update_dypre(self, obs_old, action_old, obs, action, next_obs, reward, L, step):
        """
        obs should be a list containing multimodal sensor data.
        action should be stacked and has shape [B, 3xdim].
        """
        [z, z_next, [mu, logstd]] = self.dypre.encode(obs, action, next_obs, action_condition=True)
        kl = self.dypre.kl_divergence(mu, logstd, obs_old, action_old)

        # normalize the reward before feeding it into the net
        reward = reward / self.action_repeat
        logits = self.dypre.compute_logits(z, z_next, reward)
        labels = torch.arange(logits.shape[0]).long().to(self.device)
        mi_loss = self.cross_entropy_loss(logits, labels)

        loss = mi_loss + self.omega_dypre_loss * kl

        self.encoder_optimizer.zero_grad()
        self.cpc_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.cpc_optimizer.step()
        if step % self.log_interval == 0:
            L.log('train/dypre_loss', loss, step)
            L.log('train/mi', -1.0 * mi_loss, step)
            L.log('train/kl', kl, step)

    def update(self, replay_buffer, L, step):
        if self.encoder_type == 'pixel':
            obs_chunk, action_chunk, reward_chunk, next_obs_chunk, not_done_chunk = replay_buffer.sample_consecutive(
                self.time_step)

            "set timestep"
            obs_old, obs = obs_chunk
            action_old, action = action_chunk
            _, reward = reward_chunk
            _, next_obs = next_obs_chunk
            _, not_done = not_done_chunk

        else:
            obs, action, reward, next_obs, not_done = replay_buffer.sample_proprio()

        if step % self.log_interval == 0:
            L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )

            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

            utils.soft_update_params(
                self.dypre.predictor, self.dypre.predictor_target,
                self.encoder_tau
            )

        if step % self.cpc_update_freq == 0 and self.encoder_type == 'pixel':
            self.update_dypre(obs_old, action_old, obs, action, next_obs, reward, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )


    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )

