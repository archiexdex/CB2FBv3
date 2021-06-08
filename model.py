import numpy as np
import itertools
import torch
from torch import nn, optim
from torch.autograd import Variable
from torchvision.utils import save_image, make_grid
from tqdm import tqdm

from networks import *
from Sobel import SobelGrad
from analysis import SSIM, ssim_1d, psnr
from utils import *

class Model:
    def __init__(self, opt):

        self.device = get_device()
        self.debug = opt.debug
        self.Gcf = Generator().to(self.device)
        self.Gfc = Generator().to(self.device)
        self.Dc = Discriminator().to(self.device)
        self.Df = Discriminator().to(self.device)

        self.Gcf.apply(weights_init_normal)
        self.Gfc.apply(weights_init_normal)
        self.Dc.apply(weights_init_normal)
        self.Df.apply(weights_init_normal)

        self.g_optim = optim.Adam(itertools.chain(self.Gfc.parameters(), self.Gcf.parameters()), lr=opt.lr)
        self.dc_optim  = optim.Adam(self.Dc.parameters(),  lr=opt.lr)
        self.df_optim  = optim.Adam(self.Df.parameters(),  lr=opt.lr)

        self.patch = (1, opt.crop_size // 2 ** 4, opt.crop_size // 2 ** 4)
        self.sobel = SobelGrad(5).to(self.device)
        #self.weighted = torch.ones((5), device=self.device, requires_grad=True)
        self.weighted = Weighted(5).to(self.device)

    def train(self, dataloader):
        self._unfreeze()
        mean_msssim, mean_ssim, mean_psnr = 0, 0, 0
        mseLoss = nn.MSELoss().to(self.device)
        l1Loss  = nn.L1Loss().to(self.device)
        trange = tqdm(dataloader)
        for i, data in enumerate(trange):
            if self.debug and i > 0:
                break
            real_cb, real_fb, mask = map(lambda z: z.to(self.device), data)
            ones  = torch.ones (real_cb.shape[0], *self.patch).to(self.device)
            zeros = torch.zeros(real_cb.shape[0], *self.patch).to(self.device)
            # ------------------
            #  Train G
            # ------------------
            # Mask data
            real_cb[mask<250] = 0 
            real_fb[mask<250] = 0
            # Forward
            fake_fb,  fake_cb  = self.Gcf(real_cb), self.Gfc(real_fb)
            recov_fb, recov_cb = self.Gcf(fake_cb), self.Gfc(fake_fb)
            ident_fb, ident_cb = self.Gcf(real_fb), self.Gfc(real_cb)
            pred_fb, pred_cb = self.Df(fake_fb, real_cb), self.Dc(fake_cb, real_fb)
            # GAN Loss
            fb_gan_loss = l1Loss(pred_fb, ones)
            cb_gan_loss = l1Loss(pred_cb, ones)
            gan_loss    = fb_gan_loss + cb_gan_loss
            # Cycle Loss
            fb_cycle_loss = l1Loss(recov_fb, real_fb)
            cb_cycle_loss = l1Loss(recov_cb, real_cb)
            cycle_loss    = fb_cycle_loss + cb_cycle_loss
            # Identity Loss
            fb_ident_loss = l1Loss(ident_fb, real_fb)
            cb_ident_loss = l1Loss(ident_cb, real_cb)
            ident_loss    = fb_ident_loss + cb_ident_loss
            # Content Loss
            #fb_content_loss = mseLoss(fake_fb[mask>250], real_fb[mask>250])
            #cb_content_loss = mseLoss(fake_cb[mask>250], real_cb[mask>250])
            fb_content_loss = l1Loss(fake_fb, real_fb)
            cb_content_loss = l1Loss(fake_cb, real_cb)
            content_loss    = fb_content_loss + cb_content_loss
            # Content Sobel
            fb_sobel_loss = self._get_sobel_loss(real_fb, fake_fb)
            cb_sobel_loss = self._get_sobel_loss(real_cb, fake_cb)
            sobel_loss    = fb_sobel_loss + cb_sobel_loss
            # Sum Loss
            #g_fb_loss = fb_gan_loss + fb_cycle_loss + fb_ident_loss + fb_content_loss + fb_sobel_loss
            #g_cb_loss = cb_gan_loss + cb_cycle_loss + cb_ident_loss + cb_content_loss + cb_sobel_loss
            #g_loss = g_fb_loss + g_cb_loss
            #g_loss = gan_loss + cycle_loss + ident_loss + sobel_loss + content_loss
            g_loss = torch.cat((gan_loss.unsqueeze(0), cycle_loss.unsqueeze(0), ident_loss.unsqueeze(0), sobel_loss.unsqueeze(0), content_loss.unsqueeze(0)), -1)
            g_loss = self.weighted(g_loss)
            # Backward
            self.g_optim.zero_grad()
            g_loss.backward()
            self.g_optim.step()
            # ------------------
            # Train Df
            # ------------------
            # Forward
            pred_real = self.Df(real_fb, real_cb)
            pred_fake = self.Df(fake_fb.detach(), real_cb)
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
            pred_real = self.Dc(real_cb, real_fb)
            pred_fake = self.Dc(fake_cb.detach(), real_fb)
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
                psnr_loss   = psnr(fake_fb, real_fb).item()
                msssim_loss = SSIM(fake_fb, real_fb).item()
                ssim_loss   = ssim_1d(fake_fb[mask>250], real_fb[mask>250]).item()
            # Record 
            mean_msssim += msssim_loss
            mean_ssim   += ssim_loss
            mean_psnr   += psnr_loss
            trange.set_description(f"train msssim: {mean_msssim / (i+1): .4f}, ssim: {mean_ssim / (i+1): .4f}, psnr: {mean_psnr / (i+1): .2f} \
gan: {gan_loss: .3e}, cycle: {cycle_loss: .3e}, ident: {ident_loss: .3e}, sobel: {sobel_loss: .3e}, content: {content_loss: .3e}")
        mean_msssim /= len(dataloader)
        mean_ssim   /= len(dataloader)
        mean_psnr   /= len(dataloader)
        return (mean_msssim, mean_ssim, mean_psnr)

    
    def test(self, dataloader):
        self._freeze()
        mean_msssim, mean_ssim, mean_psnr = 0, 0, 0
        mseLoss = nn.MSELoss().to(self.device)
        l1Loss  = nn.L1Loss().to(self.device)
        trange = tqdm(dataloader)
        for i, data in enumerate(trange):
            if self.debug and i > 0:
                break
            real_cb, real_fb, mask = map(lambda z: z.to(self.device), data)

            real_cb[mask<250] = 0
            real_fb[mask<250] = 0
            fake_fb = self.Gcf(real_cb)

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

    def sample_image(self, opt, dataset):
        id_list = [125, 547, 1093]
        for idx in id_list:
            data = dataset.__getitem__(idx)
            real_cb, real_fb, mask = map(lambda z: torch.unsqueeze(z, 0).to(self.device), data)
            fake_fb = self.Gcf(real_cb)
            real_cb, real_fb, fake_fb = map(lambda z: z.squeeze(0), [real_cb, real_fb, fake_fb])
            img_sample = make_grid([real_cb, fake_fb, real_fb])
            save_image(img_sample, f"{opt.rst_dir}/{opt.no}/{idx}.png")

    def save(self, path, mode="best"):
        torch.save(self.Gcf.state_dict(), os.path.join(path, f"{mode}_gcf.cpt"))
        torch.save(self.Gfc.state_dict(), os.path.join(path, f"{mode}_gfc.cpt"))
        torch.save(self.Df.state_dict(), os.path.join(path,  f"{mode}_df.cpt"))
        torch.save(self.Dc.state_dict(), os.path.join(path,  f"{mode}_dc.cpt"))

    def load(self, opt, mode="best"):
        self.Gcf.load_state_dict(torch.load(os.path.join(opt.cpt_dir, f"{opt.no}/{mode}_gcf.cpt")))
        self.Gfc.load_state_dict(torch.load(os.path.join(opt.cpt_dir, f"{opt.no}/{mode}_gfc.cpt")))
        self.Df.load_state_dict(torch.load(os.path.join(opt.cpt_dir,  f"{opt.no}/{mode}_df.cpt")))
        self.Dc.load_state_dict(torch.load(os.path.join(opt.cpt_dir,  f"{opt.no}/{mode}_dc.cpt")))

    def _get_sobel_loss(self, real, fake):
        imgx, imgy = self.sobel(fake-real)
        # sb_loss = torch.mean(imgx*imgx) + torch.mean(imgy*imgy)
        sb_loss = torch.mean(torch.abs(imgx)) + torch.mean(torch.abs(imgy))
        return sb_loss

    def _unfreeze(self):
        self.Gcf.train()
        self.Gfc.train()
        self.Df.train()
        self.Dc.train()

    def _freeze(self):
        self.Gcf.eval()
        self.Gfc.eval()
        self.Df.eval()
        self.Dc.eval()
