


def update_lr(optimizer, new_lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr

def update_linear_schedule(step, max_steps, start_lr, end_lr):
    """Decreases the learning rate linearly"""
    return start_lr - ((start_lr - end_lr) * (step / float(max_steps)))

def get_shape_from_obs_space(obs_space):
    if obs_space.__class__.__name__ == 'Box':
        if len(obs_space.shape) > 1:
            raise NotImplementedError("Multi-dimensional observation space not supported.")
        obs_shape = obs_space.shape[0]
    elif obs_space.__class__.__name__ == 'list':
        obs_shape = obs_space
    else:
        raise NotImplementedError
    return obs_shape

def torch2numpy(x):
    return x.detach().cpu().numpy()

def huber_loss(e, d):
    a = (abs(e) <= d).float()
    b = (abs(e) > d).float()
    return a*e**2/2 + b*d*(abs(e)-d/2)