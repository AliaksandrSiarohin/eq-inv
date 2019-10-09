import os
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms as T
from tqdm import tqdm

from ab_dataset import ABDataset
from network import Generator, Discriminator
from sync_batchnorm import DataParallelWithCallback
from util import make_coordinate_grid


class Transform:
    def __init__(self, bs, **kwargs):
        noise = torch.normal(mean=0, std=kwargs['sigma_affine'] * torch.ones([bs, 2, 3]))
        self.theta = noise + torch.eye(2, 3).view(1, 2, 3)

        if 'flips' in kwargs and kwargs['flips']:
            flips = (torch.randint(2, size=tuple(self.theta.shape)).type(self.theta.type()) - 0.5) * 2
            self.theta = self.theta * flips

        self.bs = bs

        if ('sigma_tps' in kwargs) and ('points_tps' in kwargs):
            self.tps = True
            self.control_points = make_coordinate_grid((kwargs['points_tps'], kwargs['points_tps']), type=noise.type())
            self.control_points = self.control_points.unsqueeze(0)
            self.control_params = torch.normal(mean=0,
                                               std=kwargs['sigma_tps'] * torch.ones([bs, 1, kwargs['points_tps'] ** 2]))
        else:
            self.tps = False

    def transform_frame(self, frame):
        grid = make_coordinate_grid(frame.shape[2:], type=frame.type()).unsqueeze(0)
        grid = grid.view(1, frame.shape[2] * frame.shape[3], 2)
        grid = self.warp_coordinates(grid).view(self.bs, frame.shape[2], frame.shape[3], 2)
        return F.grid_sample(frame, grid, padding_mode="reflection")

    def warp_coordinates(self, coordinates):
        theta = self.theta.type(coordinates.type())
        theta = theta.unsqueeze(1)
        transformed = torch.matmul(theta[:, :, :, :2], coordinates.unsqueeze(-1)) + theta[:, :, :, 2:]
        transformed = transformed.squeeze(-1)

        if self.tps:
            control_points = self.control_points.type(coordinates.type())
            control_params = self.control_params.type(coordinates.type())
            distances = coordinates.view(coordinates.shape[0], -1, 1, 2) - control_points.view(1, 1, -1, 2)
            distances = torch.abs(distances).sum(-1)

            result = distances ** 2
            result = result * torch.log(distances + 1e-6)
            result = result * control_params
            result = result.sum(dim=2).view(self.bs, coordinates.shape[1], 1)
            transformed = transformed + result

        return transformed


def normalized_l1(a, b):
    a = a.view(a.shape[0], -1)
    b = b.view(b.shape[0], -1)

    a = a / torch.norm(a, dim=1, keepdim=True)
    b = b / torch.norm(b, dim=1, keepdim=True)

    return torch.abs(a - b).sum()


class Trainer:
    def __init__(self, logger, checkpoint, device_ids, config):
        self.BtoA = config['cycle_loss_weight'] != 0
        self.config = config
        self.logger = logger
        self.device_ids = device_ids

        self.restore(checkpoint)

        print("Generator...")
        print(self.generatorB)

        print("Discriminator...")
        print(self.discriminatorB)

        transform = list()
        transform.append(T.Resize(config['load_size']))
        transform.append(T.RandomCrop(config['crop_size']))
        transform.append(T.ToTensor())
        transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
        transform = T.Compose(transform)

        self.dataset = ABDataset(config['root_dir'], partition='train', transform=transform)

    def restore(self, checkpoint):
        self.epoch = 0

        self.generatorB = Generator(**self.config['generator_params']).cuda()
        self.generatorB = DataParallelWithCallback(self.generatorB, device_ids=self.device_ids)
        self.optimizer_generatorB = torch.optim.Adam(self.generatorB.parameters(),
                                                     lr=self.config['lr_generator'], betas=(0, 0.9))

        self.discriminatorB = Discriminator(**self.config['discriminator_params']).cuda()
        self.discriminatorB = DataParallelWithCallback(self.discriminatorB, device_ids=self.device_ids)
        self.optimizer_discriminatorB = torch.optim.Adam(self.discriminatorB.parameters(),
                                                         lr=self.config['lr_discriminator'], betas=(0, 0.9))

        if self.BtoA:
            self.generatorA = Generator(**self.config['generator_params']).cuda()
            self.generatorA = DataParallelWithCallback(self.generatorA, device_ids=self.device_ids)
            self.optimizer_generatorA = torch.optim.Adam(self.generatorA.parameters(),
                                                         lr=self.config['lr_generator'], betas=(0, 0.9))

            self.discriminatorA = Discriminator(**self.config['discriminator_params']).cuda()
            self.discriminatorA = DataParallelWithCallback(self.discriminatorA, device_ids=self.device_ids)
            self.optimizer_discriminatorA = torch.optim.Adam(self.discriminatorA.parameters(),
                                                             lr=self.config['lr_discriminator'], betas=(0, 0.9))

        if checkpoint is not None:
            data = torch.load(checkpoint)
            for key, value in data.items():
                if key == 'epoch':
                    self.epoch = value
                else:
                    self.__dict__[key].load_state_dict(value)

        lr_lambda = lambda epoch: min(1, 2 - 2 * epoch / self.config['num_epochs'])
        self.scheduler_generatorB = torch.optim.lr_scheduler.LambdaLR(self.optimizer_generatorB, lr_lambda,
                                                                      last_epoch=self.epoch - 1)
        self.scheduler_discriminatorB = torch.optim.lr_scheduler.LambdaLR(self.optimizer_discriminatorB, lr_lambda,
                                                                          last_epoch=self.epoch - 1)

        if self.BtoA:
            self.scheduler_generatorA = torch.optim.lr_scheduler.LambdaLR(self.optimizer_generatorA, lr_lambda,
                                                                          last_epoch=self.epoch - 1)
            self.scheduler_discriminatorA = torch.optim.lr_scheduler.LambdaLR(self.optimizer_discriminatorA, lr_lambda,
                                                                              last_epoch=self.epoch - 1)

    def save(self):
        state_dict = {'epoch': self.epoch,
                      'generatorB': self.generatorB.state_dict(),
                      'optimizer_generatorB': self.optimizer_generatorB.state_dict(),
                      'discriminatorB': self.discriminatorB.state_dict(),
                      'optimizer_discriminatorB': self.optimizer_discriminatorB.state_dict()}

        if self.BtoA:
            state_dict.update({
                'generatorA': self.generatorA.state_dict(),
                'optimizer_generatorA': self.optimizer_generatorA.state_dict(),
                'discriminatorA': self.discriminatorA.state_dict(),
                'optimizer_discriminatorA': self.optimizer_discriminatorA.state_dict()
            })

        torch.save(state_dict, os.path.join(self.logger.log_dir, 'cpk.pth'))

    def train(self):
        np.random.seed(0)
        loader = DataLoader(self.dataset, batch_size=self.config['bs'], shuffle=False,
                            drop_last=True, num_workers=4)
        images_fixed = None

        for self.epoch in tqdm(range(self.epoch, self.config['num_epochs'])):
            loss_dict = defaultdict(lambda: 0.0)
            iteration_count = 1
            for inp in tqdm(loader):
                images_A = inp['A'].cuda()
                images_B = inp['B'].cuda()

                if images_fixed is None:
                    images_fixed = {'A': images_A, 'B': images_B}
                    transform_fixed = Transform(images_A.shape[0], **self.config['transform_params'])

                if self.config['identity_loss_weight'] != 0:
                    images_trg = self.generatorB(images_B, source=False)
                    identity_loss = torch.abs(images_trg - images_B).mean()
                    identity_loss = self.config['identity_loss_weight'] * identity_loss
                    identity_loss.backward()

                    loss_dict['identity_loss_B'] += identity_loss.detach().cpu().numpy()

                if self.config['identity_loss_weight'] != 0 and self.BtoA:
                    images_trg = self.generatorA(images_A, source=False)
                    identity_loss = torch.abs(images_trg - images_A).mean()
                    identity_loss = self.config['identity_loss_weight'] * identity_loss
                    identity_loss.backward()

                    loss_dict['identity_loss_A'] += identity_loss.detach().cpu().numpy()

                generator_loss = 0
                images_generatedB = self.generatorB(images_A, source=True)
                logits = self.discriminatorB(images_generatedB)
                adversarial_loss = ((1 - logits) ** 2).mean()
                adversarial_loss = self.config['adversarial_loss_weight'] * adversarial_loss
                generator_loss += adversarial_loss
                loss_dict['adversarial_loss_B'] += adversarial_loss.detach().cpu().numpy()

                if self.BtoA:
                    images_generatedA = self.generatorA(images_B, source=True)
                    logits = self.discriminatorA(images_generatedA)
                    adversarial_loss = ((1 - logits) ** 2).mean()
                    adversarial_loss = self.config['adversarial_loss_weight'] * adversarial_loss
                    generator_loss += adversarial_loss
                    loss_dict['adversarial_loss_A'] += adversarial_loss.detach().cpu().numpy()

                if self.config['equivariance_loss_weight_generator'] != 0:
                    transform = Transform(images_generatedB.shape[0], **self.config['transform_params'])
                    images_A_transformed = transform.transform_frame(images_A)
                    loss = normalized_l1(self.generatorB(images_A_transformed, source=True), images_generatedB)
                    loss = self.config['equivariance_loss_weight_generator'] * loss
                    generator_loss += loss

                    loss_dict['equivariance_loss_weight_generator_B'] += loss.detach().cpu().numpy()

                if self.config['equivariance_loss_weight_generator'] != 0 and self.BtoA:
                    transform = Transform(images_generatedA.shape[0], **self.config['transform_params'])
                    images_B_transformed = transform.transform_frame(images_B)
                    loss = normalized_l1(self.generatorA(images_B_transformed, source=True), images_generatedA)
                    loss = self.config['equivariance_loss_weight_generator'] * loss
                    generator_loss += loss

                    loss_dict['equivariance_loss_weight_generator_A'] += loss.detach().cpu().numpy()

                if self.BtoA and self.config['cycle_loss_weight'] != 0 and self.BtoA:
                    images_cycled = self.generatorA(images_generatedB, source=True)
                    cycle_loss = torch.abs(images_cycled - images_A).mean()
                    cycle_loss = self.config['cycle_loss_weight'] * cycle_loss
                    generator_loss += cycle_loss
                    loss_dict['cycle_loss_B'] += cycle_loss.detach().cpu().numpy()

                    images_cycled = self.generatorB(images_generatedA, source=True)
                    cycle_loss = torch.abs(images_cycled - images_B).mean()
                    cycle_loss = self.config['cycle_loss_weight'] * cycle_loss
                    generator_loss += cycle_loss
                    loss_dict['cycle_loss_A'] += cycle_loss.detach().cpu().numpy()

                generator_loss.backward()

                self.optimizer_generatorB.step()
                self.optimizer_generatorB.zero_grad()
                self.optimizer_discriminatorB.zero_grad()

                if self.BtoA:
                    self.optimizer_generatorA.step()
                    self.optimizer_generatorA.zero_grad()
                    self.optimizer_discriminatorA.zero_grad()

                logits_fake = self.discriminatorB(images_generatedB.detach())
                logits_real = self.discriminatorB(images_B)
                loss_fake = (logits_fake ** 2).mean()
                loss_real = ((1 - logits_real) ** 2).mean()

                loss_dict['fake_loss_B'] += loss_fake.detach().cpu().numpy()
                loss_dict['real_loss_B'] += loss_real.detach().cpu().numpy()

                discriminator_loss = (loss_fake + loss_real)

                if self.config['equivariance_loss_weight_discriminator'] != 0:
                    images_join = torch.cat([images_generatedB.detach(), images_B])
                    logits_join = torch.cat([logits_fake, logits_real])

                    transform = Transform(images_join.shape[0], **self.config['transform_params'])
                    images_transformed = transform.transform_frame(images_join)
                    loss = normalized_l1(self.discriminatorB(images_transformed), logits_join)

                    loss = self.config['equivariance_loss_weight_discriminator'] * loss
                    discriminator_loss += loss
                    loss_dict['equivariance_loss_weight_discriminator_B'] += loss.detach().cpu().numpy()

                discriminator_loss.backward()

                self.optimizer_discriminatorB.step()
                self.optimizer_discriminatorB.zero_grad()
                self.optimizer_generatorB.zero_grad()

                loss_dict['equivariance_loss_weight_generator_B'] += loss.detach().cpu().numpy()

                if self.BtoA:
                    logits_fake = self.discriminatorA(images_generatedA.detach())
                    logits_real = self.discriminatorA(images_A)
                    loss_fake = (logits_fake ** 2).mean()
                    loss_real = ((1 - logits_real) ** 2).mean()

                    loss_dict['fake_loss_A'] += loss_fake.detach().cpu().numpy()
                    loss_dict['real_loss_A'] += loss_real.detach().cpu().numpy()

                    discriminator_loss = (loss_fake + loss_real)

                    if self.config['equivariance_loss_weight_discriminator'] != 0:
                        images_join = torch.cat([images_generatedA.detach(), images_A])
                        logits_join = torch.cat([logits_fake, logits_real])

                        transform = Transform(images_join.shape[0], **self.config['transform_params'])
                        images_transformed = transform.transform_frame(images_join)
                        loss = normalized_l1(self.discriminatorA(images_transformed), logits_join)

                        loss = self.config['equivariance_loss_weight_discriminator'] * loss
                        discriminator_loss += loss
                        loss_dict['equivariance_loss_weight_discriminator_B'] += loss.detach().cpu().numpy()

                    discriminator_loss.backward()
                    self.optimizer_discriminatorA.step()
                    self.optimizer_discriminatorA.zero_grad()
                    self.optimizer_generatorA.zero_grad()

            iteration_count += 1

        with torch.no_grad():
            if not self.BtoA:
                self.generatorB.eval()
                transformed = transform_fixed.transform_frame(images_fixed['A'])
                self.logger.save_images(self.epoch, images_fixed['A'],
                                        self.generatorB(images_fixed['A'], source=True),
                                        transformed, self.generatorB(transformed, source=True))
                self.generatorB.train()
            else:
                self.generatorA.eval()
                self.generatorB.eval()

                images_generatedB = self.generatorB(images_fixed['A'], source=True)
                images_generatedA = self.generatorA(images_fixed['B'], source=True)

                transformed = transform_fixed.transform_frame(images_fixed['A'])
                self.logger.save_images(self.epoch,
                                        images_fixed['A'], images_generatedB,
                                        transformed, self.generatorB(transformed, source=True),
                                        self.generatorA(images_generatedB, source=True), images_fixed['B'],
                                        images_generatedA, self.generatorB(images_generatedA, source=True))

                self.generatorA.train()
                self.generatorB.train()

        self.scheduler_generatorB.step()
        self.scheduler_discriminatorB.step()
        if self.BtoA:
            self.scheduler_generatorA.step()
            self.scheduler_discriminatorA.step()

        save_dict = {key: value / iteration_count for key, value in loss_dict.items()}
        save_dict['lr'] = self.optimizer_generatorB.param_groups[0]['lr']

        self.logger.log(self.epoch, save_dict)
        self.save()
