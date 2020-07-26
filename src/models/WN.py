import torch
import torch.nn as nn

class WN_VGGLike(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        
        pool_idx = 0
        block_idx = 0
        current_hw = 32
        self.conv = nn.Sequential()
        prev_filt_size = model_params['input_shape'][0]
        for filt_size in model_params['filters']:
            self.conv.add_module(f'conv_{block_idx}', nn.utils.weight_norm(nn.Conv2d(prev_filt_size,
                                                                                     filt_size,
                                                                                     kernel_size=(3,3),
                                                                                     padding=(1,1),
                                                                                     stride=(1,1)
                                                                                    )
                                                                          )
                                )
            self.conv.add_module(f'actv_{block_idx}', nn.ELU(model_params['elu_alpha']))
            block_idx += 1
            if pool_idx < len(model_params['pool_positions']):
                if block_idx == model_params['pool_positions'][pool_idx]:
                    self.conv.add_module(f'maxpool_{block_idx}', nn.MaxPool2d((model_params['pool_factor'], model_params['pool_factor'])))
                    pool_idx += 1
                    block_idx += 1
                    current_hw *= 1/2
                    current_hw = int(current_hw)
            prev_filt_size = filt_size
            
        head_size = current_hw**2 * model_params['filters'][-1]
        self.head = nn.Sequential()
        self.head.add_module('lin', nn.utils.weight_norm(nn.Linear(head_size, head_size)))
        self.head.add_module('clf', nn.utils.weight_norm(nn.Linear(head_size, model_params['classes_nb'])))
    
        
    
    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.shape[0], -1)
        x = self.head(x)
        return x