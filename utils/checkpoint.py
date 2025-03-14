# checkpoint.py

import torch


def load_checkpoint(checkpoint_path, model_t, model_s, model_m, optimizer_t, optimizer_s, optimizer_m, scheduler_t,
                    scheduler_s, scheduler_m):
    try:
        checkpoint = torch.load(checkpoint_path)

        model_t.load_state_dict(checkpoint['model_t_state_dict'])
        model_s.load_state_dict(checkpoint['model_s_state_dict'])
        model_m.load_state_dict(checkpoint['model_m_state_dict'])

        optimizer_t.load_state_dict(checkpoint['optimizer_t_state_dict'])
        optimizer_s.load_state_dict(checkpoint['optimizer_s_state_dict'])
        optimizer_m.load_state_dict(checkpoint['optimizer_m_state_dict'])

        scheduler_t.load_state_dict(checkpoint['scheduler_t_state_dict'])
        scheduler_s.load_state_dict(checkpoint['scheduler_s_state_dict'])
        scheduler_m.load_state_dict(checkpoint['scheduler_m_state_dict'])

        model_t.to('cuda')
        model_s.to('cuda')
        model_m.to('cuda')

        for state in optimizer_t.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        for state in optimizer_s.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        for state in optimizer_m.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        return checkpoint['epoch']
    except Exception as e:
        print(f"Error loading checkpoint: {e}")
        return 0

def load_state_dict(state_dict_path, model):
    try:
        state_dict = torch.load(state_dict_path)
        model.load_state_dict(state_dict)
        model.to('cuda')
    except Exception as e:
        print(f"Error loading state dict: {e}")
