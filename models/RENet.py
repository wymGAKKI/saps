import torch
import torch.nn as nn
from torch.nn.init import kaiming_normal_
from . import model_utils

class FeatExtractor(nn.Module):
    def __init__(self, batchNorm=True, c_in=3, other={}):
        super(FeatExtractor, self).__init__()
        # batchNorm=True
        self.other = other
        self.conv1 = model_utils.conv(batchNorm, c_in, 64,  k=3, stride=1, pad=1)
        self.conv2 = model_utils.conv(batchNorm, 64,   128, k=3, stride=2, pad=1)
        self.conv3 = model_utils.conv(batchNorm, 128,  128, k=3, stride=1, pad=1)
        self.conv4 = model_utils.conv(batchNorm, 128,  256, k=3, stride=2, pad=1)
        self.conv5 = model_utils.conv(batchNorm, 256,  256, k=3, stride=1, pad=1)
        self.conv6 = model_utils.deconv(256, 128)
        self.conv7 = model_utils.conv(batchNorm, 128, 128, k=3, stride=1, pad=1)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out_feat = self.conv7(out)
        n, c, h, w = out_feat.data.shape
        out_feat   = out_feat.view(-1)
        # out_feature: torch.Size([946688])
        return out_feat, [n, c, h, w]

class Regressor(nn.Module):
    def __init__(self, batchNorm=True, other={}): 
        super(Regressor, self).__init__()
        self.other   = other
        self.deconv1 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv2 = model_utils.conv(batchNorm, 128, 128,  k=3, stride=1, pad=1)
        self.deconv3 = model_utils.deconv(128, 64)
        self.est_reflec = self._make_output(64, 3, k=3, stride=1, pad=1)
        self.other   = other

    def _make_output(self, cin, cout, k=3, stride=1, pad=1):
        return nn.Sequential(
            nn.Conv2d(cin, cout, kernel_size=k, stride=stride, padding=pad, bias=False))

    def forward(self, x, shape):
        x      = x.view(shape[0], shape[1], shape[2], shape[3])
        out    = self.deconv1(x)
        out    = self.deconv2(out)
        out    = self.deconv3(out)
        reflectance = self.est_reflec(out)
        reflectance = torch.nn.functional.normalize(reflectance, 2, 1)
        return reflectance

class RENet(nn.Module):
    def __init__(self, batchNorm=False, c_in=3, other={}):
        super(RENet, self).__init__()
        self.extractor = FeatExtractor(batchNorm, c_in, other)
        self.regressor = Regressor(batchNorm, other)
        self.c_in      = c_in
        self.fuse_type = fuse_type
        self.other = other

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                kaiming_normal_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

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
    
    def reconInputs(self, x):
        imgs = torch.split(x[0], 3, 1)
        idx = 1
        if self.other['in_light']: idx += 1
        if self.other['in_mask']:  idx += 1
        dirs = torch.split(x[idx]['dirs'], x[0].shape[0], 0)
        ints = torch.split(x[idx]['intens'], 3, 1)
        # print("dirs:", dirs[0].shape) input_img_num (batch, 3)
        # print("ints:", ints[0].shape) input_img_num (batch, 3)
        s2_inputs = {}
        lights = []
        
        for i in range(len(imgs)):
            n, c, h, w = imgs[i].shape
            lights.append(dirs[i].expand_as(imgs[i]) if dirs[i].dim() == 4 else dirs[i].view(n, -1, 1, 1).expand_as(imgs[i]))
            
        #     img   = imgs[i].contiguous().view(n * c, h * w)
        #     img   = torch.mm(l_int, img).view(n, c, h, w)
        light = torch.cat(lights, 1)

        s2_inputs['lights'] = light
        s2_inputs['ints'] = ints
        return s2_inputs

    def forward(self, x):
        inputs = self.prepareInputs(x)
        feat, shape = self.extractor(inputs)
        reflectance = self.regressor(feat, shape)
        pred = {}
        pred['reflectance'] = reflectance
        return pred