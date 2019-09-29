import torch

from collections import defaultdict
from time import gmtime, strftime
import os

from torch.utils.data import DataLoader
from torchvision import transforms as T

from tqdm import tqdm

from argparse import ArgumentParser

import json

from ab_dataset import ABDataset
from network import Generator, Discriminator
from logger import Logger


def train(dataset, generatorA, discriminatorA, generatorB, discriminatorB,  logger, args):
    if args.cycle_loss_weight != 0:
        optimizer_generatorA = torch.optim.Adam(generatorA.parameters(), lr=2e-4, betas=(0, 0.9))
        optimizer_discriminatorA = torch.optim.Adam(discriminatorA.parameters(), lr=2e-4, betas=(0, 0.9))

    optimizer_generatorB = torch.optim.Adam(generatorB.parameters(), lr=2e-4, betas=(0, 0.9))
    optimizer_discriminatorB = torch.optim.Adam(discriminatorB.parameters(), lr=2e-4, betas=(0, 0.9))

    loader = DataLoader(dataset, batch_size=args.bs, shuffle=False, drop_last=True, num_workers=4)
    images_fixed = None
    for epoch in tqdm(range(args.num_epochs)):
        loss_dict = defaultdict(lambda: 0.0)
        iteration_count = 1
        for inp in tqdm(loader):
            images_A = inp['A'].cuda()
            images_B = inp['B'].cuda()

            if images_fixed is None:
                images_fixed = {'A': images_A, 'B': images_B}
                logger.save_images(epoch=epoch, generated=generatorB(images_fixed['A']), original=images_fixed['A'],
                           direction='AtoB')

                if args.cycle_loss_weight != 0:
                    logger.save_images(epoch=epoch, generated=generatorB(images_fixed['B']), original=images_fixed['B'],
                           direction='BtoA')


            if args.identity_loss_weight != 0:
                images_trg = generatorB(images_B)
                identity_loss = torch.abs(images_trg - images_B).mean()
                identity_loss = args.identity_loss_weight * identity_loss
                identity_loss.backward()

                loss_dict['identity_loss_B'] += identity_loss.detach().cpu().numpy()

            if args.identity_loss_weight != 0 and args.cycle_loss_weight != 0:
                    images_trg = generatorA(images_A)
                    identity_loss = torch.abs(images_trg - images_A).mean()
                    identity_loss = args.identity_loss_weight * identity_loss
                    identity_loss.backward()

                    loss_dict['identity_loss_A'] += identity_loss.detach().cpu().numpy()

            generator_loss = 0
            if args.adversarial_loss_weight != 0:
                images_generatedB = generatorB(images_A)
                logits = discriminatorB(images_generatedB)
                adversarial_loss = -logits.mean()
                generator_loss += adversarial_loss

                loss_dict['adversarial_loss_B'] += adversarial_loss.detach().cpu().numpy()

            if args.adversarial_loss_weight != 0 and args.cycle_loss_weight != 0:
                images_generatedA = generatorA(images_B)
                logits = discriminatorA(images_generatedA)
                adversarial_loss = -logits.mean()
                generator_loss += adversarial_loss

                loss_dict['adversarial_loss_A'] += adversarial_loss.detach().cpu().numpy()

            if args.cycle_loss_weight != 0:
                images_cycled = generatorA(images_generatedB)
                cycle_loss = torch.abs(images_cycled - images_generatedB).mean()
                generator_loss += cycle_loss
                loss_dict['cycle_loss_B'] += cycle_loss.detach().cpu().numpy()

                images_cycled = generatorB(images_generatedA)
                cycle_loss = torch.abs(images_cycled - images_generatedA).mean()
                generator_loss += cycle_loss
                loss_dict['cycle_loss_A'] += cycle_loss.detach().cpu().numpy()

            generator_loss.backward()

            optimizer_generatorB.step()
            optimizer_generatorB.zero_grad()
            optimizer_discriminatorB.zero_grad()
            if args.cycle_loss_weight != 0:
                optimizer_generatorA.step()
                optimizer_generatorA.zero_grad()
                optimizer_discriminatorA.zero_grad()

            if args.adversarial_loss_weight != 0:
                logits_fake = discriminatorB(images_generatedB.detach())
                logits_real = discriminatorB(images_B)
                loss_fake = torch.relu(1 + logits_fake).mean()
                loss_real = torch.relu(1 - logits_real).mean()

                loss_dict['fake_loss_B'] += loss_fake.detach().cpu().numpy()
                loss_dict['real_loss_B'] += loss_real.detach().cpu().numpy()

                (loss_fake + loss_real).backward()
                optimizer_discriminatorB.step()
                optimizer_discriminatorB.zero_grad()
                optimizer_generatorB.zero_grad()

            if args.adversarial_loss_weight != 0 and args.cycle_loss_weight != 0:
                logits_fake = discriminatorA(images_generatedA.detach())
                logits_real = discriminatorA(images_B)
                loss_fake = torch.relu(1 + logits_fake).mean()
                loss_real = torch.relu(1 - logits_real).mean()

                loss_dict['fake_loss_A'] += loss_fake.detach().cpu().numpy()
                loss_dict['real_loss_A'] += loss_real.detach().cpu().numpy()

                (loss_fake + loss_real).backward()
                optimizer_discriminatorA.step()
                optimizer_discriminatorA.zero_grad()
                optimizer_generatorA.zero_grad()

            iteration_count += 1

        logger.save_images(epoch=epoch, generated=generatorB(images_fixed['A']), original=images_fixed['A'],
                           direction='AtoB')

        if args.cycle_loss_weight != 0:
            logger.save_images(epoch=epoch, generated=generatorB(images_fixed['B']), original=images_fixed['B'],
                           direction='BtoA')

        logger.log(epoch, {key: value / iteration_count for key, value in loss_dict.items()})


if __name__ == "__main__":

    parser = ArgumentParser()

    # Model configuration
    parser.add_argument('--num_epochs', type=int, default=100)
    parser.add_argument('--log_dir', default='log')
    parser.add_argument('--identity_loss_weight', type=float, default=10)
    parser.add_argument('--cycle_loss_weight', type=float, default=10)

    parser.add_argument('--adversarial_loss_weight', type=float, default=1)
    parser.add_argument('--root_dir', default='../pytorch-CycleGAN-and-pix2pix/datasets/maps/')
    parser.add_argument('--bs', type=int, default=4)

    args = parser.parse_args()

    log_dir = os.path.join(args.log_dir, strftime("%d-%m-%y %H:%M:%S", gmtime()))

    args.log_dir = log_dir

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)

    with open(os.path.join(log_dir, 'config.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    transform = list()
    transform.append(T.RandomResizedCrop(256))
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = ABDataset(args.root_dir, partition='train', transform=transform)

    if args.cycle_loss_weight != 0:
        generatorA = Generator().cuda()
        discriminatorA = Discriminator().cuda()
    else:
        generatorA = None
        discriminatorA = None

    generatorB = Generator().cuda()
    discriminatorB = Discriminator().cuda()

    logger = Logger(log_dir=log_dir)

    train(dataset, generatorA, discriminatorA, generatorB, discriminatorB, logger, args)
