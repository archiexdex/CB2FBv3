import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0.0)

class Interpolate(nn.Module):
    def __init__(self, scale):
        super(Interpolate, self).__init__()
        self.scale = scale

    def forward(self, x):
        return F.interpolate(x, scale=self.scale, recompute_scale_factor=True)

class UNetDown(nn.Module):
    def __init__(self, in_size, out_size, normalize=True, dropout=0.0):
        super().__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size=4, stride=2, padding=1, bias=True)]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_size))
            #layers.append(nn.BatchNorm2d(out_size))

        layers.append(nn.LeakyReLU(0.2))
        if dropout:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetUp(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.0):
        super().__init__()
        layers = [
            nn.Conv2d(in_size, out_size, kernel_size=3, stride=1, padding=1, bias=True),
            nn.InstanceNorm2d(out_size),
            #nn.BatchNorm2d(out_size),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x, skip_input):
        x = F.interpolate(x, scale_factor=2)
        x = self.model(x)
        x = torch.cat((x, skip_input), 1)

        return x

class ResBlock(nn.Module):
    def __init__(self, n, k=3, s=1):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Conv2d(n, n, k, stride=s, padding=1),
            #nn.InstanceNorm2d(n, affine=False),
            nn.ReLU(),
            nn.Conv2d(n, n, k, stride=s, padding=1),
            #nn.InstanceNorm2d(n, affine=False)
        )

    def forward(self, x):
        return self.net(x) + x


class Generator(nn.Module):
    def __init__(self, in_dim=1, out_dim=1, hid_dim=32, n_rb=12):
        super().__init__()

        self.down1 = UNetDown(in_dim, hid_dim, normalize=False)
        self.down2 = UNetDown(hid_dim, hid_dim<<1)
        self.down3 = UNetDown(hid_dim<<1, hid_dim<<2)
        self.down4 = UNetDown(hid_dim<<2, hid_dim<<3, dropout=0.5)
        self.down5 = UNetDown(hid_dim<<3, hid_dim<<3, normalize=False, dropout=0.5)

        self.res = nn.Sequential(*list([ResBlock(hid_dim<<3) for i in range(n_rb)]))

        self.up1 = UNetUp(hid_dim<<3, hid_dim<<3, dropout=0.5)
        self.up2 = UNetUp(hid_dim<<4, hid_dim<<2)
        self.up3 = UNetUp(hid_dim<<3, hid_dim<<1)
        self.up4 = UNetUp(hid_dim<<2, hid_dim)

        self.final = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(hid_dim<<1, out_dim, 4, padding=1),
            #nn.Tanh(),
        )

        self.out0 = nn.Conv2d(hid_dim<<1, out_dim, kernel_size=3, padding=1)
        self.out1 = nn.Conv2d(hid_dim<<2, out_dim, kernel_size=3, padding=1)
        self.out2 = nn.Conv2d(hid_dim<<3, out_dim, kernel_size=3, padding=1)

    def forward(self, x):
        # U-Net generator with skip connections from encoder to decoder
        d1 = self.down1(x)
        d2 = self.down2(d1)
        d3 = self.down3(d2)
        d4 = self.down4(d3)
        d5 = self.down5(d4)
        d5 = self.res(d5)
        u1 = self.up1(d5, d4)
        u2 = self.up2(u1, d3)
        u3 = self.up3(u2, d2)
        u4 = self.up4(u3, d1)
        u5 = self.final(u4)

        u4 = self.out0(u4)
        u3 = self.out1(u3)
        u2 = self.out2(u2)

        return {0: u5,
                1: u4,
                2: u3,
                3: u2,
                }

class Discriminator(nn.Module):
    def __init__(self, in_dim=1, hid_dim=32):
        super().__init__()

        def _block(in_dim, out_dim, normalization=True):
            """Returns downsampling layers of each  block"""
            layers = [nn.Conv2d(in_dim, out_dim, 4, stride=2, padding=1)]
            if normalization:
                #layers.append(nn.InstanceNorm2d(out_dim))
                layers.append(nn.BatchNorm2d(out_dim))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *_block(in_dim,     hid_dim,    normalization=False),
            *_block(hid_dim,    hid_dim<<1, normalization=True),
            *_block(hid_dim<<1, hid_dim<<2, normalization=True),
            *_block(hid_dim<<2, hid_dim<<3, normalization=True),
            nn.ZeroPad2d((1, 0, 1, 0)),
            nn.Conv2d(hid_dim<<3, 1, 4, padding=1, bias=False)
        )

    def forward(self, img):
        return self.model(img)
    #def forward(self, img_A, img_B):
    #    # Concatenate image and condition image by dim to produce input
    #    for i, (img_a, img_b) in enumerate(zip(img_A, img_B)):
    #        img_input = torch.cat((img_A, img_B), 1)
    #    return self.model(img_input)
