import torch

from collections import defaultdict

from sync_batchnorm import DataParallelWithCallback
from torchvision import transforms as T
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

from tqdm import tqdm
from ab_dataset import ABDataset
from network import Generator, Discriminator
import os
import numpy as np

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
        transform.append(T.RandomResizedCrop(256))#, scale=(0.9, 0.9), ratio=(1, 1)))
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
            for key, value in data:
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
                adversarial_loss = -logits.mean()
                adversarial_loss = self.config['adversarial_loss_weight'] * adversarial_loss
                generator_loss += adversarial_loss
                loss_dict['adversarial_loss_B'] += adversarial_loss.detach().cpu().numpy()

                if self.BtoA:
                    images_generatedA = self.generatorA(images_B)
                    logits = self.discriminatorA(images_generatedA, source=True)
                    adversarial_loss = -logits.mean()
                    adversarial_loss = self.config['adversarial_loss_weight'] * adversarial_loss
                    generator_loss += adversarial_loss
                    loss_dict['adversarial_loss_A'] += adversarial_loss.detach().cpu().numpy()

                if self.BtoA and self.config['cycle_loss_weight'] != 0:
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
                loss_fake = torch.relu(1 + logits_fake).mean()
                loss_real = torch.relu(1 - logits_real).mean()

                loss_dict['fake_loss_B'] += loss_fake.detach().cpu().numpy()
                loss_dict['real_loss_B'] += loss_real.detach().cpu().numpy()

                (loss_fake + loss_real).backward()
                self.optimizer_discriminatorB.step()
                self.optimizer_discriminatorB.zero_grad()
                self.optimizer_generatorB.zero_grad()

                if self.BtoA:
                    logits_fake = self.discriminatorA(images_generatedA.detach())
                    logits_real = self.discriminatorA(images_A)
                    loss_fake = torch.relu(1 + logits_fake).mean()
                    loss_real = torch.relu(1 - logits_real).mean()

                    loss_dict['fake_loss_A'] += loss_fake.detach().cpu().numpy()
                    loss_dict['real_loss_A'] += loss_real.detach().cpu().numpy()

                    (loss_fake + loss_real).backward()
                    self.optimizer_discriminatorA.step()
                    self.optimizer_discriminatorA.zero_grad()
                    self.optimizer_generatorA.zero_grad()

                iteration_count += 1

            with torch.no_grad():
                if not self.BtoA:
                    self.generatorB.eval()
                    self.logger.save_images(self.epoch, images_fixed['A'], self.generatorB(images_fixed['A'], source=True),
                                            images_fixed['A'].permute(0, 1, 3, 2),
                                            self.generatorB(images_fixed['A'].permute(0, 1, 3, 2), source=True))
                    self.generatorB.train()
                else:
                    self.generatorA.eval()
                    self.generatorB.eval()
 
                    images_generatedB = self.generatorB(images_fixed['A'], source=True)
                    images_generatedA = self.generatorA(images_fixed['B'], source=True)
                    self.logger.save_images(self.epoch,
                                            images_fixed['A'], images_generatedB,
                                            images_fixed['A'].permute(0, 1, 3, 2), self.generatorB(images_fixed['A'].permute(0, 1, 3, 2), source=True),
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
