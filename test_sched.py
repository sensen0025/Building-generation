import torch, math

# Simulate: fresh scheduler from NEW config (T_max=400000), then load_state_dict
# from OLD checkpoint scheduler (T_max=200000, base 5e-5), then step to 386000.
opt = torch.optim.AdamW([torch.zeros(1)], lr=5e-5)
sched_new = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=400000, eta_min=0.0)
# old checkpoint state (as we saw: T_max 200000, last_epoch 164000, base_lrs [5e-5])
old_state = {
    'T_max': 200000, 'eta_min': 0.0, 'base_lrs': [5e-5], 'last_epoch': 164000,
    'verbose': False, '_step_count': 164000, '_get_lr_called_within_step': False,
    '_last_lr': [3.9e-6],
}
try:
    sched_new.load_state_dict(old_state)
    print('after load_state_dict: T_max =', sched_new.T_max, 'last_epoch =', sched_new.last_epoch)
    # advance from 164000 to 386000, printing lr at key points
    for target in [164001, 200001, 230001, 266001, 300001, 340001, 386001]:
        while sched_new.last_epoch < target:
            sched_new.step()
        print(f'step {target}: lr = {sched_new.get_last_lr()[0]:.4e}  (T_max={sched_new.T_max})')
except Exception as e:
    print('ERR', e)
