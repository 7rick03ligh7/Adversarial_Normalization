import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class AdversarialNorm2d(nn.Module):
    def __init__(self, num_features, model_params):
        super().__init__()
        self.f_num = np.prod(num_features)

        self.clf = nn.Sequential()
        self.clf.add_module('conv1', nn.Conv2d(self.f_num, model_params['advers_f_num'], 3))
        self.clf.add_module('conv1_act', nn.LeakyReLU(0.1, inplace=True))
        self.clf.add_module('conv2', nn.Conv2d(model_params['advers_f_num'], model_params['advers_f_num'], 3))
        self.clf.add_module('conv2_act', nn.LeakyReLU(0.1, inplace=True))
        self.clf.add_module('pool', nn.AdaptiveAvgPool2d(1))
        self.head = nn.Linear(model_params['advers_f_num'], 1)
        
        
    def forward(self, x):
        
        x = self.clf(x)
        x = self.head(x.squeeze(2).squeeze(2))
        return x
        

class Adversarial_VGGLike(nn.Module):
    def __init__(self, model_params):
        super().__init__()
        self.elu_alpha = model_params['elu_alpha']
        
        self.conv1 = nn.Conv2d(3, model_params['filters'][0], kernel_size=(3,3), padding=(0,0), stride=(1,1))
        self.conv2 = nn.Conv2d(model_params['filters'][0], model_params['filters'][1], kernel_size=(3,3), padding=(0,0), stride=(1,1))
        self.conv3 = nn.Conv2d(model_params['filters'][1], model_params['filters'][2], kernel_size=(3,3), padding=(0,0), stride=(1,1))
        self.conv4 = nn.Conv2d(model_params['filters'][2], model_params['filters'][3], kernel_size=(3,3), padding=(0,0), stride=(1,1))
        self.conv5 = nn.Conv2d(model_params['filters'][3], model_params['filters'][4], kernel_size=(3,3), padding=(0,0), stride=(1,1))
        self.conv6 = nn.Conv2d(model_params['filters'][4], model_params['filters'][5], kernel_size=(3,3), padding=(0,0), stride=(1,1))

        self.adv_norm1 = AdversarialNorm2d((8), model_params)
        self.adv_norm2 = AdversarialNorm2d((12), model_params)
        self.adv_norm3 = AdversarialNorm2d((20), model_params)
        self.adv_norm4 = AdversarialNorm2d((28), model_params)
        
        
        self.clf = nn.Sequential()
        self.clf.add_module(
            'lin', nn.Linear(model_params['filters'][-1], model_params['filters'][-1])
            )
        self.clf.add_module(
            'lin_actv',
            nn.ELU(model_params['elu_alpha'])
            )
        self.clf.add_module(
            'clf', nn.Linear(model_params['filters'][-1], model_params['classes_nb'])
            )
    
    
    def forward(self, x, typeof_forward=0, d_count=None):
        """
            typeof_forward (int):
                0 - forward without adversarial (usefull for validation model)
                1 - forward only generator
                2 - forward only adversarial
                3 - forward everything
            
        """

        if typeof_forward == 0:
            x = F.elu(self.conv1(x), alpha=self.elu_alpha)
            x = F.elu(self.conv2(x), alpha=self.elu_alpha)
            x = F.max_pool2d(x, 2)
            x = F.elu(self.conv3(x), alpha=self.elu_alpha)
            x = F.elu(self.conv4(x), alpha=self.elu_alpha)
            x = F.max_pool2d(x, 2)
            x = F.elu(self.conv5(x), alpha=self.elu_alpha)
            x = F.elu(self.conv6(x), alpha=self.elu_alpha)
            x = x.view(x.shape[0], -1)
            x = self.clf(x)
            
            return x

        elif typeof_forward == 1:
            x = self.conv1(x)
            x1 = F.elu(x, alpha=self.elu_alpha)

            x = self.conv2(x1)
            x2 = F.elu(x, alpha=self.elu_alpha)

            x = F.max_pool2d(x2, 2)

            x = self.conv3(x)
            x3 = F.elu(x, alpha=self.elu_alpha)

            x = self.conv4(x3)
            x4 = F.elu(x, alpha=self.elu_alpha)  

            return [x1, x2, x3, x4]
        
        elif typeof_forward == 2:
            x1, x2, x3, x4 = x

            z1 = self.adv_norm1(x1)
            if d_count == 1:
                return z1
            z2 = self.adv_norm2(x2)
            if d_count == 2:
                return z2
            z3 = self.adv_norm3(x3)
            if d_count == 3:
                return z3
            z4 = self.adv_norm4(x4)
            if d_count == 4:
                return z4
            
            return [z1, z2, z3, z4]

        elif typeof_forward == 3:
            x = self.conv1(x)
            z1 = self.adv_norm1(x)
            x = F.elu(x, alpha=self.elu_alpha)

            x = self.conv2(x)
            z2 = self.adv_norm2(x)
            x = F.elu(x, alpha=self.elu_alpha)

            x = F.max_pool2d(x, 2)

            x = self.conv3(x)
            z3 = self.adv_norm3(x)
            x = F.elu(x, alpha=self.elu_alpha)

            x = self.conv4(x)
            z4 = self.adv_norm4(x)
            x = F.elu(x, alpha=self.elu_alpha)

            x = F.max_pool2d(x, 2)

            x = self.conv5(x)
            x = F.elu(x, alpha=self.elu_alpha)

            x = self.conv6(x)
            x = F.elu(x, alpha=self.elu_alpha)

            x = x.view(x.shape[0], -1)        
            x = self.clf(x)
            
            return x, [z1, z2, z3, z4]