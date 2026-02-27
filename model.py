import torch
import torch.nn as nn


# -----------------------------
#   Attention (CBAM) modules
# -----------------------------
class ChannelAttention(nn.Module):
    """
    CBAM Channel Attention (Woo et al., ECCV 2018).
    Uses both global average pooling and global max pooling.
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.mlp = nn.Sequential(
            nn.Conv2d(channels, hidden, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, channels, kernel_size=1, bias=False),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (B,C,1,1)
        avg = torch.mean(x, dim=(2, 3), keepdim=True)
        mx = torch.amax(x, dim=(2, 3), keepdim=True)
        att = self.mlp(avg) + self.mlp(mx)
        return x * self.sigmoid(att)


class SpatialAttention(nn.Module):
    """
    CBAM Spatial Attention (Woo et al., ECCV 2018).
    Uses channel-wise average and max projection, then a conv.
    """
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        assert kernel_size in (3, 7)
        padding = 3 if kernel_size == 7 else 1
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # (B,1,H,W)
        avg = torch.mean(x, dim=1, keepdim=True)
        mx, _ = torch.max(x, dim=1, keepdim=True)
        att = self.conv(torch.cat([avg, mx], dim=1))
        return x * self.sigmoid(att)


class CBAM(nn.Module):
    """
    Convolutional Block Attention Module (CBAM):
    channel attention -> spatial attention
    """
    def __init__(self, channels: int, reduction: int = 16, spatial_kernel: int = 7):
        super().__init__()
        self.ca = ChannelAttention(channels, reduction=reduction)
        self.sa = SpatialAttention(kernel_size=spatial_kernel)

    def forward(self, x):
        x = self.ca(x)
        x = self.sa(x)
        return x


# -----------------------------
#   Building blocks
# -----------------------------
class ConvBNAct(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, s=1, p=1, act=True):
        super().__init__()
        layers = [
            nn.Conv2d(in_ch, out_ch, kernel_size=k, stride=s, padding=p, bias=False),
            nn.BatchNorm2d(out_ch, momentum=0.01),
        ]
        if act:
            layers.append(nn.LeakyReLU(inplace=True))
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResidualCBAMBlock(nn.Module):
    """
    A simple residual block + optional CBAM.
    Not "Attention U-Net" (no attention gates on skips);
    it injects lightweight CBAM attention inside the conv block.
    """
    def __init__(self, in_ch, out_ch, dropout=0.0, use_cbam=True):
        super().__init__()
        self.conv1 = ConvBNAct(in_ch, out_ch, k=3, s=1, p=1, act=True)
        self.dropout = nn.Dropout2d(dropout, inplace=True) if dropout and dropout > 0 else nn.Identity()
        self.conv2 = ConvBNAct(out_ch, out_ch, k=3, s=1, p=1, act=False)

        self.cbam = CBAM(out_ch) if use_cbam else nn.Identity()

        if in_ch != out_ch:
            self.shortcut = ConvBNAct(in_ch, out_ch, k=1, s=1, p=0, act=False)
        else:
            self.shortcut = nn.Identity()

        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.cbam(out)
        out = out + self.shortcut(x)
        return self.act(out)


class UpBlock(nn.Module):
    
    def __init__(self, in_ch, skip_ch, out_ch, dropout=0.0, use_cbam=True):
        super().__init__()
        self.up = nn.ConvTranspose2d(in_ch, in_ch, kernel_size=2, stride=2)
        self.block = ResidualCBAMBlock(in_ch + skip_ch, out_ch, dropout=dropout, use_cbam=use_cbam)

    def forward(self, x, skip):
        x = self.up(x)
        # shapes should match because training uses fixed patch size divisible by 2^3
        x = torch.cat([skip, x], dim=1)
        return self.block(x)


# -----------------------------
#   Original model kept (optional)
# -----------------------------
class FCUnetFactorized(nn.Module):
    """
    Original Factorized U-net for Retinal Vessel Segmentation (as provided in the repo).
    Kept here so you can switch back easily if needed.
    """
    def __init__(self, **kwargs):
        super().__init__()
        dropout_rate = kwargs['dropout rate']

        filters_0 = kwargs['base filters']
        filters_1 = 2 * filters_0
        filters_2 = 4 * filters_0
        filters_3 = 8 * filters_0

        class UpsampleBlock(nn.Module):
            def __init__(self, in_channels, cat_channels, out_channels):
                super().__init__()
                self.bottleneck = nn.Sequential(
                    nn.Conv2d(in_channels + cat_channels, out_channels, (3, 3), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(out_channels, momentum=0.01),
                    nn.LeakyReLU(inplace=True)
                )
                self.upsample = nn.Sequential(
                    nn.ConvTranspose2d(in_channels, in_channels, (2, 2), stride=(2, 2)),
                    nn.LeakyReLU(inplace=True),
                )
            def forward(self, x):
                upsample, concat = x
                upsample = self.upsample(upsample)
                return self.bottleneck(torch.cat([concat, upsample], 1))

        class FactorizedBlock(nn.Module):
            def __init__(self, in_channels, inner_channels):
                super().__init__()
                self.conv_0 = nn.Sequential(
                    nn.Conv2d(in_channels, inner_channels, (3, 3), padding=(1, 1), bias=False),
                    nn.BatchNorm2d(inner_channels, momentum=0.01),
                    nn.LeakyReLU(inplace=True)
                )
                self.conv_1_3 = nn.Conv2d(inner_channels, inner_channels, (1, 3), padding=(0, 1), bias=False)
                self.conv_3_1 = nn.Conv2d(inner_channels, inner_channels, (3, 1), padding=(1, 0), bias=False)
                self.norm = nn.Sequential(
                    nn.BatchNorm2d(2 * inner_channels, momentum=0.01),
                    nn.LeakyReLU(inplace=True)
                )
            def forward(self, x):
                out = self.conv_0(x)
                return self.norm(torch.cat([self.conv_1_3(out), self.conv_3_1(out)], 1))

        # Encoder
        self.block_0_0 = nn.Sequential(
            nn.Conv2d(3, filters_0, (3, 3), padding=(1, 1)),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_0, filters_0 // 2)
        )
        self.block_1_0 = nn.Sequential(
            nn.Conv2d(filters_0, filters_1, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_1, filters_1 // 2)
        )
        self.block_2_0 = nn.Sequential(
            nn.Conv2d(filters_1, filters_2, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_2, filters_2 // 2)
        )
        self.block_3_0 = nn.Sequential(
            nn.Conv2d(filters_2, filters_3, kernel_size=(2, 2), stride=(2, 2)),
            nn.LeakyReLU(inplace=True),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_3, filters_3 // 2)
        )

        # Decoder
        self.block_2_1 = nn.Sequential(
            UpsampleBlock(filters_3, filters_2, filters_2),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_2, filters_2 // 2)
        )
        self.block_1_1 = nn.Sequential(
            UpsampleBlock(filters_2, filters_1, filters_1),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_1, filters_1 // 2)
        )
        self.block_0_1 = nn.Sequential(
            UpsampleBlock(filters_1, filters_0, filters_0),
            nn.Dropout2d(dropout_rate, inplace=True),
            FactorizedBlock(filters_0, filters_0 // 2),
            nn.Conv2d(filters_0, 1, (3, 3), padding=(1, 1))
        )

    def forward(self, inputs):
        out_0 = self.block_0_0(inputs)
        out_1 = self.block_1_0(out_0)
        out_2 = self.block_2_0(out_1)
        out_3 = self.block_3_0(out_2)
        out_2 = self.block_2_1([out_3, out_2])
        out_1 = self.block_1_1([out_2, out_1])
        return self.block_0_1([out_1, out_0])


# -----------------------------
#   New model: CBAM-ResUNet (lightweight)
# -----------------------------
class CBAMResUNet(nn.Module):
    """
    Residual U-Net + CBAM (channel+spatial attention) inside residual blocks.
    - Not "Attention U-Net" gates; attention is embedded per-block (CBAM).
    - Designed to be drop-in compatible with the rest of this repo:
      input: (B,3,H,W) -> output logits: (B,1,H,W)
    """
    def __init__(self, **kwargs):
        super().__init__()
        dropout = float(kwargs.get('dropout rate', 0.0))
        base = int(kwargs.get('base filters', 16))

        f0, f1, f2, f3 = base, 2*base, 4*base, 8*base

        # Encoder
        self.enc0 = ResidualCBAMBlock(3,  f0, dropout=dropout, use_cbam=True)
        self.down1 = nn.Conv2d(f0, f1, kernel_size=2, stride=2)
        self.enc1 = ResidualCBAMBlock(f1, f1, dropout=dropout, use_cbam=True)

        self.down2 = nn.Conv2d(f1, f2, kernel_size=2, stride=2)
        self.enc2 = ResidualCBAMBlock(f2, f2, dropout=dropout, use_cbam=True)

        self.down3 = nn.Conv2d(f2, f3, kernel_size=2, stride=2)
        self.bottleneck = ResidualCBAMBlock(f3, f3, dropout=dropout, use_cbam=True)

        # Decoder
        self.up2 = UpBlock(f3, f2, f2, dropout=dropout, use_cbam=True)
        self.up1 = UpBlock(f2, f1, f1, dropout=dropout, use_cbam=True)
        self.up0 = UpBlock(f1, f0, f0, dropout=dropout, use_cbam=True)

        self.head = nn.Conv2d(f0, 1, kernel_size=1)

        self.act = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # Encoder
        s0 = self.enc0(x)           # (B,f0,H,W)
        x  = self.act(self.down1(s0))
        s1 = self.enc1(x)           # (B,f1,H/2,W/2)

        x  = self.act(self.down2(s1))
        s2 = self.enc2(x)           # (B,f2,H/4,W/4)

        x  = self.act(self.down3(s2))
        x  = self.bottleneck(x)     # (B,f3,H/8,W/8)

        # Decoder
        x = self.up2(x, s2)         # (B,f2,H/4,W/4)
        x = self.up1(x, s1)         # (B,f1,H/2,W/2)
        x = self.up0(x, s0)         # (B,f0,H,W)

        return self.head(x)         # logits (B,1,H,W)


FCUnet = CBAMResUNet