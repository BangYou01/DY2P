import torch
import torch.nn as nn

def tie_weights(src, trg):
    assert type(src) == type(trg)
    trg.weight = src.weight
    trg.bias = src.bias

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

# for 84 x 84 inputs
OUT_DIM = {2: 39, 4: 35, 6: 31}
# for 64 x 64 inputs
OUT_DIM_64 = {2: 29, 4: 25, 6: 21}


class PixelEncoder(nn.Module):
    """Convolutional encoder of pixels observations.
    Input: tensor
        RGB image: [B,9, 84, 84]
        depth image: [B,3, 84, 84]
    Output: list
        mu: [B, mu_dim]
        logstd: [B, std_dim]

    Todolist:
        1. how to determine feat_dim_depth and feat_dim_rgb
    """
    def __init__(self, obs_shape, feat_dim_rgb, num_layers=2, num_filters=32,output_logits=False, modality='rgb'):
        super().__init__()

        assert len(obs_shape) == 3
        self.obs_shape = obs_shape
        self.num_layers = num_layers
        self.feature_dim = feat_dim_rgb

        self.convs = nn.ModuleList(
            [nn.Conv2d(obs_shape[0], num_filters, 3, stride=2)]
        )
        for i in range(num_layers - 1):
            self.convs.append(nn.Conv2d(num_filters, num_filters, 3, stride=1))

        out_dim = OUT_DIM_64[num_layers] if obs_shape[-1] == 64 else OUT_DIM[num_layers] 
        self.fc = nn.Linear(num_filters * out_dim * out_dim, self.feature_dim)

        self.net = nn.Sequential(
            nn.Linear(self.feature_dim, 1024), nn.ReLU(),
            nn.Linear(1024, 2 * self.feature_dim),
        )
        self.ln = nn.LayerNorm(self.feature_dim)

        self.outputs = dict()
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd, output_mean=False):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)

        if output_mean:
            output = mu
        else:
            output = mu + eps * std
        return output

    def forward_conv(self, obs):
        obs = obs / 255.
        self.outputs['obs'] = obs

        conv = torch.relu(self.convs[0](obs))
        self.outputs['conv1'] = conv

        for i in range(1, self.num_layers):
            conv = torch.relu(self.convs[i](conv))
            self.outputs['conv%s' % (i + 1)] = conv

        h = conv.view(conv.size(0), -1)
        return h

    def forward(self, obs, detach=False):
        h = self.forward_conv(obs)

        if detach:
            h = h.detach()

        h_fc = self.fc(h)
        self.outputs['fc'] = h_fc

        h_norm = self.ln(h_fc)
        self.outputs['ln'] = h_norm

        if self.output_logits:
            h_out = h_norm
        else:
            h_out = h_fc

        return h_out

    def copy_conv_weights_from(self, source):
        """Tie convolutional layers"""
        # only tie conv layers
        for i in range(self.num_layers):
            tie_weights(src=source.convs[i], trg=self.convs[i])

    def log(self, L, step, log_freq):
        if step % log_freq != 0:
            return

        for k, v in self.outputs.items():
            L.log_histogram('train_encoder/%s_hist' % k, v, step)
            if len(v.shape) > 2:
                L.log_image('train_encoder/%s_img' % k, v[0], step)

        for i in range(self.num_layers):
            L.log_param('train_encoder/conv%s' % (i + 1), self.convs[i], step)
        L.log_param('train_encoder/fc', self.fc, step)
        L.log_param('train_encoder/ln', self.ln, step)

class Touch_encoder(nn.Module):
    """
    MLP encoder for handling touch data

    Input: tensor
        touch_data: [B, 6]
    Output: list
        mu: [B, mu_dim]
        logstd: [B, std_dim]
    """
    def __init__(self, touch_shape, feature_num=10, output_logits=False):
        super().__init__()

        self.touch_shape = touch_shape
        self.feature_num = feature_num
        self.output_logits = output_logits

        self.net = nn.Sequential(
            nn.Linear(self.touch_shape[0],512), nn.ReLU(),
            nn.Linear(512, self.feature_num)
        )

        self.ln = nn.LayerNorm(self.feature_num)
        self.output_logits = output_logits

    def reparameterize(self, mu, logstd, output_mean=False):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)

        if output_mean:
            output = mu
        else:
            output = mu + eps * std
        return output

    def forward(self, touch_data, detach=False):

        output = self.net(touch_data)

        if detach:
            output = output.detach()

        output_norm = self.ln(output)

        if self.output_logits:
            output = output_norm
        else:
            output = output

        return output


class Action_encoder(nn.Module):
    """
    MLP encoder of actions.

    Input: tensor
        actions: [B, dim]
    Output: tensor
        output: [B, z_dim]
    """
    def __init__(self, action_shape, feature_num=10, output_logits=False):
        super().__init__()

        self.action_shape = action_shape

        self.feature_num = feature_num
        self.output_logits = output_logits

        self.net = nn.Sequential(
            nn.Linear(self.action_shape,512), nn.ReLU(),
            nn.Linear(512,self.feature_num)
        )

        self.ln = nn.LayerNorm(self.feature_num)
        self.output_logits = output_logits

    def forward(self, actions, detach=False):
        output = self.net(actions)

        if detach:
            output = output.detach()

        output_norm = self.ln(output)

        if self.output_logits:
            output = output_norm
        else:
            output = torch.tanh(output_norm)
        return output

class Transition_model(nn.Module):

    def __init__(self,latent_action_shape, z_dim, output_feature_num=50, num_layers=2, output_logits=False):
        super().__init__()
        self.latent_action_shape = latent_action_shape
        self.latent_feature_shape = z_dim
        self.output_feature_num = output_feature_num
        self.num_layers = num_layers
        self.output_logits = output_logits

        self.net = nn.Sequential(
            nn.Linear(self.latent_action_shape + self.latent_feature_shape, 1024), nn.ReLU(),
            nn.Linear(1024,1024), nn.ReLU(), nn.Linear(1024, 2 * self.output_feature_num),
        )

        self.ln = nn.LayerNorm(2 * self.output_feature_num)

    def forward(self, latent_obs, latent_actions, condition = True, detach = False):
        obs_actions = torch.cat([latent_obs, latent_actions], dim=1)
        output = self.net(obs_actions)

        if detach:
            output = output.detach()

        output_norm = self.ln(output)

        if self.output_logits:
            output = output_norm
        else:
            output = output

        mu, logstd = output.chunk(2, dim=-1)
        #mu, logstd = _activation([mu, logstd])
        state = self.reparameterize(mu, logstd)

        return [mu, logstd, state]

    def reparameterize(self, mu, logstd, output_mean=False):
        std = torch.exp(logstd)
        eps = torch.randn_like(std)

        if output_mean:
            output = mu
        else:
            output = mu + eps * std
        return output

class Fusion(nn.Module):
    """
    Fusion of multi-modal data

    Input:
        mu = {'rgb': mu_img, 'depth': mu_depth, 'touch': mu_touch} mu_xxx: [B, dim]
        std = {'rgb': std_img, 'depth': std_depth, 'touch': std_touch}
    return:
        mu: [B, dim]
        std: [B, dim]
    """

    def __init__(self, args):
        super().__init__()
        self.args = args
        self.set_fusion_functions()
        self.subsets = {'rgb': ['rgb'], 'depth': ['depth'], 'touch': ['touch'], 
                        'depth_rgb': ['depth', 'rgb'], 'rgb_touch': ['rgb', 'touch'], 'depth_touch': ['depth', 'touch'], 
                        'rgb_depth_touch': ['rgb', 'depth', 'touch']}
        
    def set_fusion_functions(self):
        # Switch the fusion method, `PoE`, `MoE` or `MoPoE` 
        if self.args.modality_poe:
            self.modality_fusion = self.poe_fusion
            self.fusion_condition = self.fusion_condition_poe
        elif self.args.modality_moe:
            self.modality_fusion = self.moe_fusion
            self.fusion_condition = self.fusion_condition_moe
        elif self.args.joint_elbo:
            self.modality_fusion = self.poe_fusion
            self.fusion_condition = self.fusion_condition_joint

    def poe_fusion(self, mus, logvars, weights=None):
        # TODO: dim
        if (self.args.modality_poe or mus.shape[0] == 3):
            # DO NOT know why it should cat zeros.
            num_samples = mus[0].shape[0]
            feat_dim = mus[0].shape[1]
            mus = torch.cat((mus, torch.zeros(1, num_samples, feat_dim).to(self.args.device)), dim=0)
            # mus = [[mu00, mu01, ...], [mu10, mu11, ...], [mu20, mu21, ...], [0, 0, ...]]
            logvars = torch.cat((logvars, torch.zeros(1, num_samples, feat_dim).to(self.args.device)), dim=0)
        mu_poe, logvar_poe = self.poe(mus, logvars)
        return [mu_poe, logvar_poe]

    def poe(self, mu, logvar, eps=1e-8):
        var = torch.exp(logvar) + eps
        # precision of i-th Gaussian expert at point x
        T = 1. / var
        pd_mu = torch.sum(mu * T, dim=0) / torch.sum(T, dim=0)
        pd_var = 1. / torch.sum(T, dim=0)
        pd_logvar = torch.log(pd_var + eps)
        return pd_mu, pd_logvar

    def moe_fusion(self, mus, logvars, weights=None):
        weights = reweight_weights(weights)
        mu_moe, logvar_moe = mixture_component_selection(self.args, mus, logvars, weights)
        return [mu_moe, logvar_moe]

    def fusion_condition_moe(self, subset, input_batch=None):
        if len(subset) == 1:
            return True
        else:
            return False

    def fusion_condition_poe(self, subset, input_batch=None):
        if len(subset) == len(input_batch.keys()):
            return True
        else:
            return False

    def fusion_condition_joint(self, subset, input_batch=None):
        return True

    def forward(self, mu, logvar):
        '''
        mu = {'rgb': mu_img, 'depth': mu_depth, 'touch': mu_touch}
        std = {'rgb': std_img, 'depth': std_depth, 'touch': std_touch}
        '''
        # For the original MoPoE, input_batch = {'img': xxx, 'text': xxx}
        #   batch_size = 256 = B
        #   input_batch['img'].shape = torch.Size([256, 3, 64, 64])
        #   input_batch['text'].shape = torch.Size([256, 256, 71])
        #   num_samples = self.args.batch_size 256
        #   subsets = {'': [], 'img': [Img object], 'img_text': [Img object  Text object], 'text': [Text object]}
        #   enc_mods mu logvar:  torch.Size([256, 20])
        #   mus_subset:  torch.Size([m, 256, 20])  m equals to the num of modalities in subsets (e.g. img m = 1, img_text m = 2)
        #   s_mu: torch.Size([256, 20]), s_mu is obtained from fusion of mus_subset (using PoE)
        #   mus: torch.Size([n, 256, 20]) n equals to the length of subsets
        #   joint_mu: torch.Size([256, 20]), joint_mu is obtained from fusion of mus (using MoE)

        mus = torch.Tensor().to(self.args.device)
        logvars = torch.Tensor().to(self.args.device)
        
        # subsets = {'': [], 'rgb': [ImageRgb], 'depth': [ImageDepth]. 'touch': [Touch],
        #            'depth_rgb': [ImageDepth, ImageRgb], 'rgb_touch': [ImageRgb, Touch]
        #            'depth_touch': [ImageDepth, Touch], 'depth_rgb_touch': [ImageDepth, ImageRgb, Touch]}
        for s_key in self.subsets:
            mods = self.subsets[s_key]
            mus_subset = torch.Tensor().to(self.args.device)
            logvars_subset = torch.Tensor().to(self.args.device)
            mods_avail = True
            for mod in mods:
                if mod in mu.keys():
                    # mu[mod].unsqueeze(0) from [B, dim] to [1, B, dim]
                    mus_subset = torch.cat((mus_subset, mu[mod].unsqueeze(0)), dim=0)
                    logvars_subset = torch.cat((logvars_subset, logvar[mod].unsqueeze(0)), dim=0)
                else:
                    mods_avail = False
            if mods_avail:
                    weights_subset = ((1/float(len(mus_subset)))*
                                      torch.ones(len(mus_subset)).to(self.args.device))
                    s_mu, s_logvar = self.modality_fusion(mus_subset,
                                                          logvars_subset,
                                                          weights_subset)
                    if self.fusion_condition(mods, mu):
                        mus = torch.cat((mus, s_mu.unsqueeze(0)), dim=0)
                        logvars = torch.cat((logvars, s_logvar.unsqueeze(0)), dim=0)
        weights = (1/float(mus.shape[0]))*torch.ones(mus.shape[0]).to(self.args.device)
        joint_mu, joint_logvar = self.moe_fusion(mus, logvars, weights)
        return joint_mu, joint_logvar


def reweight_weights(w):
    w = w / w.sum()
    return w

def mixture_component_selection(flags, mus, logvars, w_modalities=None):
    # if num_components = 7, num_samples = 20
    # idx_start = [0, 2, 6, 8, 10, 12]
    # idx_end = [2, 4, 6, 8, 12, 20]
    num_components = mus.shape[0]
    num_samples = mus.shape[1]
    idx_start = []
    idx_end = []
    for k in range(0, num_components):
        if k == 0:
            i_start = 0
        else:
            i_start = int(idx_end[k-1])
        if k == w_modalities.shape[0]-1:
            i_end = num_samples
        else:
            i_end = i_start + int(torch.floor(num_samples*w_modalities[k]))
        idx_start.append(i_start)
        idx_end.append(i_end)
    idx_end[-1] = num_samples
    mu_sel = torch.cat([mus[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
    logvar_sel = torch.cat([logvars[k, idx_start[k]:idx_end[k], :] for k in range(w_modalities.shape[0])])
    return [mu_sel, logvar_sel]


_AVAILABLE_ENCODERS = {'pixel': PixelEncoder}

def make_encoder(
    encoder_type, obs_shape, feature_dim, num_layers, num_filters, output_logits=False
):
    assert encoder_type in _AVAILABLE_ENCODERS
    return _AVAILABLE_ENCODERS[encoder_type](
        obs_shape, feature_dim, num_layers, num_filters, output_logits
    )
