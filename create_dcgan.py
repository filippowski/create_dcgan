import config as cfg
import torch
from netG import _netG
from netD import _netD
from util import print_params, weights_init
import torchvision.utils as vutils
from torch.autograd import Variable
import torch.optim as optim

#######################################################################################################################

class DCGAN:

    def __init__(self):
        self.ngpu = int(cfg.ngpu)
        self.nz   = int(cfg.nz)
        self.ngf  = int(cfg.ngf)
        self.ndf  = int(cfg.ndf)
        self.nc   = int(cfg.nc)

        self.netG = cfg.netG
        self.netD = cfg.netD

        self.nepoch      = cfg.nepoch
        self.batchSize   = cfg.batchSize
        self.imageSize   = cfg.imageSize
        self.real_label  = cfg.real_label
        self.fake_label  = cfg.fake_label

        self.input       = Variable(torch.FloatTensor(self.batchSize, 3, self.imageSize, self.imageSize))
        self.noise       = Variable(torch.FloatTensor(self.batchSize, self.nz, 1, 1))
        self.fixed_noise = Variable(torch.FloatTensor(self.batchSize, self.nz, 1, 1).normal_(0, 1))
        self.label       = Variable(torch.FloatTensor(self.batchSize))
        self.output      = None

        self.lr          = cfg.lr
        self.beta1       = cfg.beta1

        self.loss        = cfg.loss

        self.optimizerG  = None
        self.optimizerD  = None

        self.dataloader  = cfg.dataloader
        self.outf        = cfg.outf
        self.cuda        = cfg.cuda

    # CREATE DCGAN
    def create(self):

        print('\n\nGenerator net: ')
        netG = _netG(self.ngpu, self.nz, self.ngf, self.nc)
        netG.apply(weights_init)
        if self.netG != '':
            netG.load_state_dict(torch.load(self.netG))
        print_params(netG)

        print('\n\nDiscriminator net: ')
        netD = _netD(self.ngpu, self.nz, self.ndf, self.nc)
        netD.apply(weights_init)
        if self.netD != '':
            netD.load_state_dict(torch.load(self.netD))
        print_params(netD)

        self.netG = netG
        self.netD = netD


    def setup(self):
        # create G and D nets
        self.create()

        # if cuda available, run with it
        self.run_cuda()

        # setup optimizer
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=self.lr, betas=(self.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=self.lr, betas=(self.beta1, 0.999))


    def run_cuda(self):
        if self.cuda:
            self.netD.cuda()
            self.netG.cuda()
            self.loss.cuda()
            self.input, self.label = self.input.cuda(), self.label.cuda()
            self.noise, self.fixed_noise = self.noise.cuda(), self.fixed_noise.cuda()

    # TRAIN DCGAN
    def train(self):

        for epoch in range(self.nepoch):
            for i, data in enumerate(self.dataloader, 0):

                ############################
                # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
                ###########################
                # train with real
                self.netD.zero_grad()
                real_cpu, _ = data
                batch_size = real_cpu.size(0)
                self.input.data.resize_(real_cpu.size()).copy_(real_cpu)
                self.label.data.resize_(batch_size).fill_(self.real_label)

                self.output = self.netD(self.input)
                errD_real   = self.loss(self.output, self.label)
                errD_real.backward()
                D_x = self.output.data.mean()

                # train with fake
                self.noise.data.resize_(batch_size, self.nz, 1, 1)
                self.noise.data.normal_(0, 1)
                fake = self.netG(self.noise)
                self.label.data.fill_(self.fake_label)
                self.output = self.netD(fake.detach())
                errD_fake = self.loss(self.output, self.label)
                errD_fake.backward()
                D_G_z1 = self.output.data.mean()
                errD = errD_real + errD_fake
                self.optimizerD.step()

                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                self.netG.zero_grad()
                self.label.data.fill_(self.real_label)  # fake labels are real for generator cost
                self.output = self.netD(fake)
                errG = self.loss(self.output, self.label)
                errG.backward()
                D_G_z2 = self.output.data.mean()
                self.optimizerG.step()

                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, self.nepoch, i, len(self.dataloader),
                         errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
                if i % 100 == 0:
                    vutils.save_image(real_cpu,
                                      '%s/real_samples.png' % self.outf,
                                      normalize=True)
                    fake = netG(self.fixed_noise)
                    vutils.save_image(fake.data,
                                      '%s/fake_samples_epoch_%03d.png' % (self.outf, epoch),
                                      normalize=True)

            # do checkpointing
            torch.save(self.netG.state_dict(), '%s/netG_epoch_%d.pth' % (self.outf, epoch))
            torch.save(self.netD.state_dict(), '%s/netD_epoch_%d.pth' % (self.outf, epoch))


#######################################################################################################################
if __name__ == '__main__':

    dcgan = DCGAN()

    dcgan.setup()
    dcgan.train()