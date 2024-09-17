import torch
import torch.nn as nn
import numpy as np

def layer_init(layer, std=None, bias_const=0.0):
    if std is None:
        torch.nn.init.orthogonal_(layer.weight)
    else:
        torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

def huber_replace(td_errors, kappa: float = 1.0):
    return torch.where(torch.abs(td_errors) <= kappa, 0.5 * td_errors ** 2, kappa * (torch.abs(td_errors) - 0.5 * kappa))

def calculate_quantile_huber_loss(td_errors, taus, kappa: float = 1.0):
    element_wise_huber_loss = huber_replace(td_errors, kappa)
    mask = torch.where(td_errors < 0, 1, 0).detach() # detach this
    element_wise_quantile_huber_loss = torch.abs(taus[..., None] - mask) * element_wise_huber_loss / kappa
    quantile_huber_loss = element_wise_quantile_huber_loss.sum(dim=1).mean()
    return quantile_huber_loss

# fix init
class BroNetCritic(nn.Module):
    def __init__(self, state_size, action_size, output_nodes):
        super().__init__()
        self.block1 = nn.Sequential(layer_init(nn.Linear(state_size+action_size, 512)), nn.LayerNorm(512), nn.ReLU())
        self.block2 = nn.Sequential(layer_init(nn.Linear(512, 512)), nn.LayerNorm(512), nn.ReLU())
        self.block3 = nn.Sequential(layer_init(nn.Linear(512, 512)), nn.LayerNorm(512))
        self.block4 = nn.Sequential(layer_init(nn.Linear(512, 512)), nn.LayerNorm(512), nn.ReLU())
        self.block5 = nn.Sequential(layer_init(nn.Linear(512, 512)), nn.LayerNorm(512))
        self.final_layer = layer_init(nn.Linear(512, output_nodes))
        
    def forward(self, state, action):
        x = torch.concat((state, action), dim=-1)
        x = self.block1(x)
        res = self.block2(x)
        res = self.block3(res)
        x = x + res
        res = self.block4(x)
        res = self.block5(res)
        x = res + x
        x = self.final_layer(x)
        return x
    
class BroNetCritics(nn.Module):
    def __init__(self, state_size, action_size, output_nodes):
        super().__init__()
        self.critic1 = BroNetCritic(state_size, action_size, output_nodes)
        self.critic2 = BroNetCritic(state_size, action_size, output_nodes)
        
    def forward(self, state, action):
        q1 = self.critic1(state, action)
        q2 = self.critic2(state, action)
        return q1, q2
        
class BroNetActor(nn.Module):
    def __init__(self, state_size, action_size):
        super().__init__()
        self.log_std_min = -10
        self.log_std_max = 2
        self.block1 = nn.Sequential(layer_init(nn.Linear(state_size, 256)), nn.LayerNorm(256), nn.ReLU())
        self.block2 = nn.Sequential(layer_init(nn.Linear(256, 256)), nn.LayerNorm(256), nn.ReLU())
        self.block3 = nn.Sequential(layer_init(nn.Linear(256, 256)), nn.LayerNorm(256))
        self.means = layer_init(nn.Linear(256, action_size))
        self.log_stds = layer_init(nn.Linear(256, action_size), std=0.01)
        
    def forward(self, state, temperature=1.0):
        x = self.block1(state)
        res = self.block2(x)
        res = self.block3(x)
        x = x + res
        means = self.means(x)
        log_stds = self.log_stds(x)
        log_stds = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(log_stds))
        stds = log_stds.exp() * temperature
        return means, stds
    
class BRO(nn.Module):
    def __init__(self, state_size, action_size, pessimism, learning_rate, n_quantiles, discount, device='cpu'):
        super().__init__()
        self.discount = discount
        self.pessimism = pessimism
        self.state_size, self.action_size = state_size, action_size
        self.n_quantiles = n_quantiles
        quantile_taus = torch.arange(0, n_quantiles+1) / n_quantiles
        self.quantile_taus = ((quantile_taus[1:] + quantile_taus[:-1]) / 2.0)[None, ...]
        self.action_scale = 1.0
        self.action_bias = 0.0
        self.kappa = 1.0
        self.device = device
        self.learning_rate = learning_rate
        self.reset()
        
    def reset(self):
        self.critic = BroNetCritics(self.state_size, self.action_size, self.n_quantiles).to(self.device)
        self.target_critic = BroNetCritics(self.state_size, self.action_size, self.n_quantiles).to(self.device)
        self.actor = BroNetActor(self.state_size, self.action_size).to(self.device)
        self.log_temp = torch.tensor(np.log(1.0)).to(self.device)
        self.log_temp.requires_grad = True
        self.optimizer_log_temp = torch.optim.Adam([self.log_temp], lr=self.learning_rate)
        self.optimizer_critic = torch.optim.AdamW(self.critic.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        self.optimizer_actor = torch.optim.AdamW(self.actor.parameters(), lr=self.learning_rate, weight_decay=1e-3)
        
    @property
    def temp(self):
        return self.log_temp.exp()

    def get_action_and_logprob(self, state, temperature=1.0, get_log_prob=True):
        mu, std = self.actor(state, temperature)
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.action_scale + self.action_bias
        if get_log_prob:
            log_prob = normal.log_prob(x_t)
            # Enforcing Action Bound
            log_prob -= torch.log(self.action_scale * (1 - y_t.pow(2)) + 1e-6)
            log_prob = log_prob.sum(1, keepdim=True)
            return action, log_prob
        return action

    def update_critic(self, states, actions, rewards, next_states, masks):
        with torch.no_grad():
            next_actions, next_log_probs = self.get_action_and_logprob(states)
            next_q1, next_q2 = self.target_critic(next_states, next_actions)
        q_mean = (next_q1 + next_q2) / 2
        q_uncertainty = torch.abs(next_q1 - next_q2) / 2
        next_q = q_mean - self.pessimism * q_uncertainty
        target_q = rewards[:, None, None] + self.discount * masks[:, None, None]  * next_q[:, None, :]
        target_q -= self.discount * self.temp() * masks[:, None, None] * next_log_probs[:, :, None]
        q1, q2 = self.critic(states, actions)
        td_errors1 = target_q - q1[:, :, None]
        td_errors2 = target_q - q2[:, :, None] 
        critic_loss = calculate_quantile_huber_loss(td_errors1, self.quantile_taus, kappa=self.kappa) + calculate_quantile_huber_loss(td_errors2, self.quantile_taus, kappa=self.kappa)
        self.optimizer_critic.zero_grad()
        critic_loss.backward()
        self.optimizer_critic.step()
        return {'critic_loss': critic_loss.detach().item(),
                'q_mean': q_mean.mean().detach().item(),
                'q_uncertainty': q_uncertainty.detach().item()}
    
    def update_actor(self, states):
        actions, log_probs = self.get_action_and_logprob(states)
        q1, q2 = self.critic(states, actions)
        q = (q1 + q2) / 2 - self.pessimism * torch.abs(q1 - q2) / 2
        q = q.mean(-1, keepdim=True)
        actor_loss = (self.temp() * log_probs - q).mean()
        self.optimizer_actor.zero_grad()
        actor_loss.backward()
        self.optimizer_actor.step()
        # update temp
        self.optimizer_log_temp.zero_grad()
        entropy = -log_probs.mean()
        temp_loss = (self.temp * (entropy - self.target_entropy).detach())
        temp_loss.backward()
        self.optimizer_log_temp.step()
        return {'actor_loss': actor_loss.detach().item(),
                'entropy': entropy.detach().item(),
                'temp_loss': temp_loss.detach().item()}
    
    def update_target_critic(self):
        for param, target_param in zip(self.critic.parameters(), self.target_critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
    
    def single_update(self, states, actions, rewards, next_states, masks):
        critic_info = self.update_critic(states, actions, rewards, next_states, masks)
        self.update_target_critic()
        actor_info = self.update_actor(states)
        return {**actor_info, **critic_info}
    
    def update(self, step, buffer, batch_size, replay_ratio):
        states, actions, rewards, next_states, masks = buffer.sample(batch_size)
        for i in range(replay_ratio):
            info = self.single_update(states, actions, rewards, next_states, masks)
        return info
    
'''  
critic = BroNetCritics(10, 4, 100)
actor = BroNetActor(10, 4)
agent = BRO(10, 4, 1.0 , 100, 0.99)

state = torch.ones(32, 10).float()
action = torch.ones(32, 4).float()
reward = torch.ones(32).float()
next_state = torch.ones(32, 10).float()
mask = torch.ones(32).float()
loss_c = agent.update_critic(state, action, reward, next_state, mask)
loss_a = agent.update_actor(state)

q1, q2 = critic(state, action)
means, std = actor(state)
action, log_prob = agent.get_action_and_logprob(state)
'''