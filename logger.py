import os
import torch
from util import denorm
from collections import OrderedDict
from torchvision.utils import save_image

class Logger:
    def __init__(self, log_dir, log_file='log.txt', vis_dir='train-vis'):
        self.log_file = os.path.join(log_dir, log_file)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

        self.vis_dir = os.path.join(log_dir, vis_dir)
        if not os.path.exists(self.vis_dir):
            os.makedirs(self.vis_dir)

        self.cpk_dir = log_dir

    # def save_cpk(self, epoch, generator, discriminator, generator_optimizer, discriminator_optimizer):
    #     torch.save({'epoch' : epoch, 'generator': generator, 'discriminator': discriminator,
    #                 'generator_optimizer': generator_optimizer, 'discriminator_optimizer': discriminator_optimizer},
    #                os.path.join(self.cpk_dir, str(epoch).zfill(5) + '-cpk.pth.tar'))

    def save_images(self, epoch, direction, original, generated):
        x_concat = torch.cat([original, generated], dim=2)
        sample_path = os.path.join(self.vis_dir, str(epoch).zfill(5) + direction + '.png')
        save_image(denorm(x_concat.data.cpu()), sample_path, nrow=4, padding=0)

    def log(self, epoch, scores):
        result = ''
        for key, value in OrderedDict(scores).items():
            result += '{}:{:.2f} '.format(key, value)
        with open(self.log_file, 'a') as f:
            print('{})'.format(str(epoch).zfill(3)) + result, file=f)



