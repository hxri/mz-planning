import torch
import time
import numpy as np
import torch.nn as nn
import torch.optim as optim
from .advantage_estimate import advantage_estimate

def ppo(envs, agent, config, device, writer):
    batch_size = int(config['num_envs'] * config['num_steps'])
    minibatch_size = int(batch_size // config['num_minibatches'])

    optimizer = optim.Adam(agent.parameters(), lr=config['learning_rate'], eps=1e-5)

    # intitalize zero tensors for s, a, log_prob, r, t, v
    obs = torch.zeros((config['num_steps'], config['num_envs']) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    logprobs = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    rewards = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    dones = torch.zeros((config['num_steps'], config['num_envs'])).to(device)
    values = torch.zeros((config['num_steps'], config['num_envs'])).to(device)

    global_step = 0
    start_time = time.time()

    # Reset the envs and get an observation
    next_obs = torch.Tensor(envs.reset()[0]).to(device)
    next_done = torch.zeros(config['num_envs']).to(device)
    num_updates = config['total_timesteps'] // batch_size

    # print(next_obs)

    # Start the PPO updation process
    for update in range(1, num_updates + 1):
        
        # Early stop for testing purposes or limiting the training process
        if(config['early_stop'] is not None and global_step > config['early_stop']):
            break

        # Learning rate annealing
        if config['anneal_lr']:
            frac = 1.0 - (update - 1.0) / num_updates
            lrnow = frac * config['learning_rate']
            optimizer.param_groups[0]['lr'] = lrnow

        epsilon = config['initial_epsilon']

        for step in range(0, config['num_steps']):
            global_step += 1 * config['num_envs']
            obs[step] = next_obs
            dones[step] = next_done

            # get action and value using Actor and Critic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs, epsilon)
                # print(action)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            next_obs, reward, terminated, truncated, info = envs.step(action.cpu().numpy())
            # print(np.average(reward))

            rewards[step] = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(terminated).to(device)

            # print(info)
            if('final_info' in info):
                # print(info)
                for item in info['final_info']:
                    if(item):
                        if 'episode' in item.keys():
                            # return_arr = item['episode']['r'].item()
                            print(f"global_step={global_step}, episodic_return={item['episode']['r']}")
                            writer.add_scalar("charts/episodic_return", item['episode']['r'], global_step)
                            writer.add_scalar("charts/episodic_length", item['episode']['l'], global_step)
                            break
            
            # update epsilon
            epsilon = epsilon * config['decay_parameter']
                 
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)

            advantages, returns = advantage_estimate(config, rewards, device, next_done, next_value, values, dones)
        # Flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape(-1)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(batch_size)
        clipfracs = []
        for epoch in range(config['update_epochs']):
            np.random.shuffle(b_inds)
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(
                    b_obs[mb_inds], epsilon, b_actions.long()[mb_inds]
                )
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > config['clip_coef']).float().mean().detach().cpu().numpy()]

                mb_advantages = b_advantages[mb_inds]
                if config['norm_adv']:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                
                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - config['clip_coef'], 1 + config['clip_coef'])
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if config['clip_vloss']:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) **2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds]
                        -config['clip_coef'],
                        config['clip_coef'],
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()

                loss = pg_loss + config['ent_coef'] * entropy_loss + config['vf_coef'] * v_loss

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), config['max_grad_norm'])
                optimizer.step()
            
            if config['target_kl'] is not None:
                if approx_kl > config['target_kl']:
                    break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]['lr'], global_step)
        writer.add_scalar("charts/value_loss", v_loss.item(), global_step)
        writer.add_scalar("charts/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("charts/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("charts/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("charts/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("charts/explained_variance", explained_var, global_step)
        print("SPS: ", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    return agent