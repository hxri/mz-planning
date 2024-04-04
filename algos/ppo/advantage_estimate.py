import torch

def advantage_estimate(config, rewards, device, next_done, next_value, values, dones):
    if config['gae']:
        advantages = torch.zeros_like(rewards).to(device)
        lastgaelam = 0
        for t in reversed(range(config['num_steps'])):
            if t == config['num_steps'] -1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t+1]
                nextvalues = values[t+1]
            delta = rewards[t] + config['gamma'] * nextvalues * nextnonterminal - values[t]
            advantages[t] = lastgaelam = delta + config['gamma'] * config['gae_lambda'] * nextnonterminal * lastgaelam
        returns = advantages + values
    else:
        returns = torch.zeros_like(rewards).to(device)
        for t in reversed(range(config['num_steps'])):
            if t == config['num_steps'] -1:
                nextnonterminal = 1.0 - next_done
                nextvalues = next_value
            else:
                nextnonterminal = 1.0 - dones[t+1]
                next_return = returns[t+1]
            returns[t] = rewards[t] + config['gamma'] * nextnonterminal * next_return
        advantages = returns - values

    return advantages, returns