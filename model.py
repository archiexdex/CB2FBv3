import numpy as np
import itertools
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision.utils import  make_grid
from tqdm import tqdm

from utils import *
from networks import *
from Sobel import SobelGrad
from analysis import SSIM, ssim_1d, psnr

class Model:
    def __init__(self, opt):

        self.opt = opt
        self.device = get_device()

        # Construct models
        self.models = {}
        self.models["Gcf"] = Generator().to(self.device)
        self.models["Gfc"] = Generator().to(self.device)
        self.models["Dc"]  = Discriminator().to(self.device)
        self.models["Df"]  = Discriminator().to(self.device)

        # Weight init 
        for i in self.models:
            self.models[i].apply(weights_init_normal)

        # Construct optimizer
        self.g_optim  = optim.Adam(list(self.models["Gcf"].parameters())+list(self.models["Gfc"].parameters()), lr=opt.lr)
        self.dc_optim = optim.Adam(self.models["Dc"].parameters(),  lr=opt.lr)
        self.df_optim = optim.Adam(self.models["Df"].parameters(),  lr=opt.lr)

        # Construct scheduler
        self.g_scheduler  = optim.lr_scheduler.StepLR(self.g_optim, step_size=30, gamma=0.5)
        self.dc_scheduler = optim.lr_scheduler.StepLR(self.dc_optim, step_size=30, gamma=0.5)
        self.df_scheduler = optim.lr_scheduler.StepLR(self.df_optim, step_size=30, gamma=0.5)

        self.patch = (1, opt.crop_size // 2 ** 4, opt.crop_size // 2 ** 4)
        self.sobel = SobelGrad(kernel_size=3, channel=1, device=self.device)

    def train(self, dataloader, epoch):
        self._train(True)
        mean_msssim, mean_ssim, mean_psnr = 0, 0, 0
        mseLoss = nn.MSELoss().to(self.device)
        l1Loss  = nn.L1Loss().to(self.device)
        trange = tqdm(dataloader)
        for i, data in enumerate(trange):
            if self.opt.debug and i > 0:
                break
            real_cb, real_fb, mask = map(lambda z: z.to(self.device), data)
            ones  = torch.ones (real_cb.shape[0], *self.patch).to(self.device)
            zeros = torch.zeros(real_cb.shape[0], *self.patch).to(self.device)
            _l1Loss = lambda x, y: torch.mean(torch.abs(x-y).reshape(x.shape[0], -1), -1)
            _l2Loss = lambda x, y: torch.mean((x-y).reshape(x.shape[0], -1), -1)
            # ------------------
            #  Train G
            # ------------------
            # Mask data
            real_cb[mask<250] = 0 
            real_fb[mask<250] = 0
            # Forward
            fake_fb,  fake_cb  = self.models["Gcf"](real_cb),          self.models["Gfc"](real_fb)
            recov_fb, recov_cb = self.models["Gcf"](fake_cb[0]),          self.models["Gfc"](fake_fb[0])
            ident_fb, ident_cb = self.models["Gcf"](real_fb),          self.models["Gfc"](real_cb)
            pred_fb,  pred_cb  = self.models["Df"] (fake_fb[0], real_cb), self.models["Dc"] (fake_cb[0], real_fb)
            # Recover all size
            self._interpolate(fake_fb), self._interpolate(fake_cb)
            self._interpolate(recov_fb), self._interpolate(recov_cb)
            self._interpolate(ident_fb), self._interpolate(ident_cb)
            # GAN Loss
            fb_gan_loss = l1Loss(pred_fb, ones)
            cb_gan_loss = l1Loss(pred_cb, ones)
            gan_loss    = fb_gan_loss + cb_gan_loss
            # Cycle Loss
            fb_cycle_loss =  sum([l1Loss(recov_fb[i], real_fb) for i in range(4)])
            cb_cycle_loss = sum([l1Loss(recov_cb[i], real_cb) for i in range(4)])
            cycle_loss    = fb_cycle_loss + cb_cycle_loss
            # Identity Loss
            fb_ident_loss = sum([l1Loss(ident_fb[i], real_fb) for i in range(4)])
            cb_ident_loss = sum([l1Loss(ident_cb[i], real_cb) for i in range(4)])
            ident_loss    = fb_ident_loss + cb_ident_loss
            # Content Loss
            fb_content_loss = sum([l1Loss(fake_fb[i], real_fb) for i in range(4)])
            cb_content_loss = sum([l1Loss(fake_cb[i], real_cb) for i in range(4)])
            content_loss    = fb_content_loss + cb_content_loss
            # Content Sobel
            fb_sobel_loss = sum([self._get_sobel_loss(real_fb, fake_fb[i]) for i in range(4)])
            cb_sobel_loss = sum([self._get_sobel_loss(real_cb, fake_cb[i]) for i in range(4)])
            sobel_loss    = fb_sobel_loss + cb_sobel_loss
            # Sum loss
            if epoch < self.opt.warm_epoch:
                g_loss = content_loss + sobel_loss
            else:
                g_loss = 1e-2 * gan_loss + (cycle_loss + ident_loss + sobel_loss + content_loss) / 4
            # Backward
            self.g_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()
            # ------------------
            # Train Df
            # ------------------
            if epoch >= self.opt.warm_epoch:
                # Forward
                pred_real = self.models["Df"](real_fb, real_cb)
                pred_fake = self.models["Df"](fake_fb[0].detach(), real_cb)
                loss_real = l1Loss(pred_real, ones)
                loss_fake = l1Loss(pred_fake, zeros)
                # Sum Loss
                d_loss = (loss_real + loss_fake) * 0.5
                # Backward
                self.df_optim.zero_grad()
                d_loss.backward()
                self.df_optim.step()
                # ------------------
                # Train Dc
                # ------------------
                # Forward
                pred_real = self.models["Dc"](real_cb, real_fb)
                pred_fake = self.models["Dc"](fake_cb[0].detach(), real_fb)
                loss_real = mseLoss(pred_real, ones)
                loss_fake = mseLoss(pred_fake, zeros)
                # Sum Loss
                d_loss = (loss_real + loss_fake) * 0.5
                # Backward
                self.dc_optim.zero_grad()
                d_loss.backward()
                self.dc_optim.step()

            # Analysis
            with torch.no_grad():
                psnr_loss   = psnr(fake_fb[0], real_fb).item()
                msssim_loss = SSIM(fake_fb[0], real_fb).item()
                ssim_loss   = ssim_1d(fake_fb[0][mask>250], real_fb[mask>250]).item()
            # Record 
            mean_msssim += msssim_loss
            mean_ssim   += ssim_loss
            mean_psnr   += psnr_loss
#            trange.set_description(f"train msssim: {mean_msssim / (i+1): .4f}, ssim: {mean_ssim / (i+1): .4f}, psnr: {mean_psnr / (i+1): .2f} \
#gan: {gan_loss.mean(): .3e}, cycle: {cycle_loss.mean(): .3e}, ident: {ident_loss.mean(): .3e}, sobel: {sobel_loss.mean(): .3e}, content: {content_loss.mean(): .3e}")
            trange.set_description(f"train msssim: {mean_msssim / (i+1): .4f}, ssim: {mean_ssim / (i+1): .4f}, psnr: {mean_psnr / (i+1): .2f} \
gan: {gan_loss: .3e}, cycle: {cycle_loss: .3e}, ident: {ident_loss: .3e}, sobel: {sobel_loss: .3e}, content: {content_loss: .3e}")
        mean_msssim /= len(dataloader)
        mean_ssim   /= len(dataloader)
        mean_psnr   /= len(dataloader)
        return (mean_msssim, mean_ssim, mean_psnr)

    
    def test(self, dataloader):
        self._train(True)
        mean_msssim, mean_ssim, mean_psnr = 0, 0, 0
        mseLoss = nn.MSELoss().to(self.device)
        l1Loss  = nn.L1Loss().to(self.device)
        trange = tqdm(dataloader)
        for i, data in enumerate(trange):
            if self.opt.debug and i > 0:
                break
            real_cb, real_fb, mask = map(lambda z: z.to(self.device), data)

            real_cb[mask<250] = 0
            real_fb[mask<250] = 0
            fake_fb = self.models["Gcf"](real_cb)[0]

            # Analysis
            psnr_loss   = psnr(fake_fb, real_fb).item()
            msssim_loss = SSIM(fake_fb, real_fb).item()
            ssim_loss   = ssim_1d(fake_fb[mask>250], real_fb[mask>250]).item()
            # Record 
            mean_msssim += msssim_loss
            mean_ssim   += ssim_loss
            mean_psnr   += psnr_loss
            trange.set_description(f"valid msssim: {mean_msssim / (i+1): .4f}, ssim: {mean_ssim / (i+1): .4f}, psnr: {mean_psnr / (i+1): .2f}")
        mean_msssim /= len(dataloader)
        mean_ssim   /= len(dataloader)
        mean_psnr   /= len(dataloader)
        return (mean_msssim, mean_ssim, mean_psnr)

    def scheduler_step(self, epoch):
        self.g_scheduler.step()
        if epoch >= self.opt.warm_epoch:
            self.dc_scheduler.step()
            self.df_scheduler.step()

    def sample_image(self, dataset, sample_mode="fix"):
        id_list, buf = [], {} 
        if sample_mode == "all":
            id_list = [i for i in range(len(dataset))]
        elif sample_mode == "random":
            # Notice: do not use same_seed to fix random
            for i in range(3):
                id_list.append(random.randint(0, len(dataset)-1))
        elif sample_mode == "fix":
            id_list = [125, 547, 1093]

        for idx in id_list:
            data = dataset.__getitem__(idx)
            real_cb, real_fb, mask = map(lambda z: torch.unsqueeze(z, 0).to(self.device), data)
            fake_fb = self.models["Gcf"](real_cb)[0]
            real_cb, real_fb, fake_fb, mask = map(lambda z: z.squeeze(0), [real_cb, real_fb, fake_fb, mask])
            img_sample = make_grid([real_cb, fake_fb, real_fb])
            buf[f"{idx}"] = [real_cb, real_fb, fake_fb, mask, img_sample]
        return buf

    def save(self, path, mode="best"):
        for i in self.models:
            torch.save(self.models[i].state_dict(), os.path.join(path, f"{mode}_{i}.cpt"))

    def load(self, mode="best"):
        for i in self.models:
            self.models[i].load_state_dict(torch.load(os.path.join(self.opt.cpt_dir, f"{self.opt.no}/{mode}_{i}.cpt")))
            self.models[i].to(self.device)

    def _interpolate(self, imgs):
        for i in imgs.keys():
            imgs[i] = F.interpolate(imgs[i], scale_factor=1<<i)
        return imgs

    def _get_sobel_loss(self, real, fake):
        imgx, imgy = self.sobel(fake-real)
        # sb_loss = torch.mean(imgx*imgx) + torch.mean(imgy*imgy)
        #sb_loss = torch.mean(torch.abs(imgx).reshape(imgx.shape[0], -1), -1) + torch.mean(torch.abs(imgy).reshape(imgy.shape[0], -1), -1)
        sb_loss = torch.mean(torch.abs(imgx)) + torch.mean(torch.abs(imgy))
        return sb_loss

    def _train(self, flg=False):
        for i in self.models:
            self.models[i].train(flg)
