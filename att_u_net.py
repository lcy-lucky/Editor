import torch
import torch.nn as nn
import torch.nn.functional as functional

class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)

class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out

class CBAM(nn.Module):
    def __init__(self, channel):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()

    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out

class UpAttentionBlock(nn.Module):
    def __init__(self, in_channels_x, in_channels_g, int_channels):
        super(UpAttentionBlock, self).__init__()
        self.Wx = nn.Sequential(nn.Conv2d(in_channels_x, int_channels, kernel_size = 1),
                                nn.BatchNorm2d(int_channels))
        self.Wg = nn.Sequential(nn.Conv2d(in_channels_g, int_channels, kernel_size = 1),
                                nn.BatchNorm2d(int_channels))
        self.psi = nn.Sequential(nn.Conv2d(int_channels, 1, kernel_size = 1),
                                 nn.BatchNorm2d(1),
                                 nn.Sigmoid())
    
    def forward(self, x, g):
        # apply the Wx to the skip connection
        x1 = self.Wx(x)
        # print("x1", x1.shape)
        # after applying Wg to the input, upsample to the size of the skip connection
        g1 = nn.functional.interpolate(self.Wg(g), x1.shape[2:], mode = 'bilinear', align_corners = False)
        out = self.psi(nn.ReLU()(x1 + g1))
        return out*x

def add_conv_stage(dim_in, dim_out, kernel_size=3, stride=1, padding=1, bias=True, useBN=False):
  if useBN:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.BatchNorm2d(dim_out),
      nn.LeakyReLU(0.1)
    )
  else:
    return nn.Sequential(
      nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU(),
      nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
      nn.ReLU()
    )

def upsample(ch_coarse, ch_fine):
  return nn.Sequential(
    nn.ConvTranspose2d(ch_coarse, ch_fine, 4, 2, 1, bias=False),
    nn.ReLU()
  )

class Net(nn.Module):
  def __init__(self, useBN=False):
    super(Net, self).__init__()

    self.conv1 = add_conv_stage(1, 8, useBN=useBN)
    self.conv2 = add_conv_stage(8, 16, useBN=useBN)
    self.conv3 = add_conv_stage(16, 32, useBN=useBN)
    self.conv4 = add_conv_stage(32, 64, useBN=useBN)
    self.conv5 = add_conv_stage(64, 128, useBN=useBN)

    self.cbam1 = CBAM(channel=8)
    self.cbam2 = CBAM(channel=16)
    self.cbam3 = CBAM(channel=32)
    self.cbam4 = CBAM(channel=64)

    self.upattion4 = UpAttentionBlock(in_channels_x=64, in_channels_g=128, int_channels=64)
    self.upattion3 = UpAttentionBlock(in_channels_x=32, in_channels_g=64, int_channels=32)
    self.upattion2 = UpAttentionBlock(in_channels_x=16, in_channels_g=32, int_channels=16)
    self.upattion1 = UpAttentionBlock(in_channels_x=8, in_channels_g=16, int_channels=8)

    self.conv4m = add_conv_stage(128, 64, useBN=useBN)
    self.conv3m = add_conv_stage(64, 32, useBN=useBN)
    self.conv2m = add_conv_stage(32, 16, useBN=useBN)
    self.conv1m = add_conv_stage(16, 8, useBN=useBN)

    self.conv0  = nn.Sequential(
        nn.Conv2d(8, 1, 3, 1, 1),
        nn.Sigmoid()
    )

    self.max_pool = nn.MaxPool2d(2)

    self.upsample54 = upsample(128, 64)
    self.upsample43 = upsample(64, 32)
    self.upsample32 = upsample(32, 16)
    self.upsample21 = upsample(16, 8)

    ## weight initialization
    for m in self.modules():
      if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
        if m.bias is not None:
          m.bias.data.zero_()


  def forward(self, x):
    conv1_out = self.conv1(x)
    print(conv1_out.shape)
    conv1_out = self.cbam1(conv1_out) + conv1_out
    print(conv1_out.shape)

    conv2_out = self.conv2(self.max_pool(conv1_out))
    print(conv2_out.shape)
    conv2_out = self.cbam2(conv2_out) + conv2_out
    print(conv2_out.shape)
    
    conv3_out = self.conv3(self.max_pool(conv2_out))
    print(conv3_out.shape)
    conv3_out = self.cbam3(conv3_out) + conv3_out
    print(conv3_out.shape)

    conv4_out = self.conv4(self.max_pool(conv3_out))
    print(conv4_out.shape)
    conv4_out = self.cbam4(conv4_out) + conv4_out
    print(conv4_out.shape)
    conv5_out = self.conv5(self.max_pool(conv4_out))
    print(conv5_out.shape)
    

    # conv5m_out = torch.cat((self.upsample54(conv5_out), conv4_out), 1)
    conv5m_out = torch.cat((self.upsample54(conv5_out), self.upattion4(conv4_out,conv5_out)), 1)
    print(conv5m_out.shape)
    conv4m_out = self.conv4m(conv5m_out)
    print(conv4m_out.shape)

    # conv4m_out_ = torch.cat((self.upsample43(conv4m_out), conv3_out), 1)
    conv4m_out_ = torch.cat((self.upsample43(conv4m_out), self.upattion3(conv3_out,conv4m_out)), 1)
    print(conv4m_out.shape)
    conv3m_out = self.conv3m(conv4m_out_)
    print(conv3m_out.shape)

    # conv3m_out_ = torch.cat((self.upsample32(conv3m_out), conv2_out), 1)
    conv3m_out_ = torch.cat((self.upsample32(conv3m_out), self.upattion2(conv2_out,conv3m_out)), 1)
    print(conv3m_out.shape)
    conv2m_out = self.conv2m(conv3m_out_)
    print(conv2m_out.shape)

    # conv2m_out_ = torch.cat((self.upsample21(conv2m_out), conv1_out), 1)
    conv2m_out_ = torch.cat((self.upsample21(conv2m_out), self.upattion1(conv1_out,conv2m_out)), 1)
    print(conv2m_out.shape)
    conv1m_out = self.conv1m(conv2m_out_)
    print(conv1m_out.shape)
  

    conv0_out = self.conv0(conv1m_out)
    print(conv0_out.shape)

    return conv0_out


if __name__=='__main__':
    model = Net(useBN=False)
    x = torch.randn([1,1,48,48])
    print(model(x).shape)



    