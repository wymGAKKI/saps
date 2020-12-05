import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils
#import model_utils

class skipRENet(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(skipRENet, self).__init__()
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64,   128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128,  128, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128,  256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256,  256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

        self.deconv1 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 256, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(256, 64)
        self.est_reflec = self._make_output(64 + c_in, 3, k=3, stride=1, pad=1)
        self.skip = nn.Identity()

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False),
                nn.LeakyReLU(0.1, inplace=True))
    
    def prepareInputs(self, x):
        #x = {'img': img, 'mask': mask, 'dirs': dirs, 'reflectance':reflectance}
        img = x['img']
        # lights = torch.split(x['dirs'], 3, 1)
        mask = x['mask']
        # dirs = torch.split(x['dirs'], x[0].shape[0], 0)
        # print("dirs:", dirs[0].shape)
        # print("ints:", ints[0].shape)
        inputs = torch.cat([img, mask], 1)
        return inputs

    def forward(self, x):
        input = self.prepareInputs(x)
        res1 = input
        out1 = self.conv1(input)
        out2 = self.conv2(out1)
        res2 = out2
        out3 = self.conv3(out2)
        res3 = out3
        out4 = self.conv4(out3)
        out5 = self.conv5(out4)
        out6 = self.conv6(out5)
        out7 = self.conv7(out6)
        out8 = self.deconv1(out7)

        out9    = self.deconv2(torch.cat([out8, self.skip(res3)], 1))
        out10    = self.deconv3(torch.cat([out9, self.skip(res2)], 1))
        out11 = self.est_reflec(torch.cat([out10, self.skip(res1)], 1))
        reflectance = out11
        pred_r = {}
        pred_r['reflectance'] = reflectance
        return pred_r

if __name__ == '__main__':
    module = newRENet(True, 32*3, {})
    input = torch.randn((8, 96, 256, 256))
    out = module(input)
    #print(input.shape)