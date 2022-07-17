from torch import nn


class Autoencoder(nn.Module):
    def __init__(self):
        super(Autoencoder, self).__init__()
        self.conv1_1 = self._conv_unit(3, 64)
        self.conv1_2 = self._conv_unit(64, 64)
        self.pool1 = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv2_1 = self._conv_unit(64, 128)
        self.conv2_2 = self._conv_unit(128, 128)
        self.pool2 = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv3_1 = self._conv_unit(128, 256)
        self.conv3_2 = self._conv_unit(256, 256)
        self.conv3_3 = self._conv_unit(256, 256)
        self.pool3 = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv4_1 = self._conv_unit(256, 512)
        self.conv4_2 = self._conv_unit(512, 512)
        self.conv4_3 = self._conv_unit(512, 512)
        self.pool4 = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv5_1 = self._conv_unit(512, 512)
        self.conv5_2 = self._conv_unit(512, 512)
        self.conv5_3 = self._conv_unit(512, 512)
        self.pool5 = nn.MaxPool2d(2, 2, return_indices=True)
        self.conv6 = self._conv_unit(512, 512)

        self.unpool5 = nn.MaxUnpool2d(2, 2)
        self.deconv5_1 = self._deconv_unit(512, 512)
        self.deconv5_2 = self._deconv_unit(512, 512)
        self.deconv5_3 = self._deconv_unit(512, 512)
        self.unpool4 = nn.MaxUnpool2d(2, 2)
        self.deconv4_1 = self._deconv_unit(512, 512)
        self.deconv4_2 = self._deconv_unit(512, 512)
        self.deconv4_3 = self._deconv_unit(512, 256)
        self.unpool3 = nn.MaxUnpool2d(2, 2)
        self.deconv3_1 = self._deconv_unit(256, 256)
        self.deconv3_2 = self._deconv_unit(256, 256)
        self.deconv3_3 = self._deconv_unit(256, 128)
        self.unpool2 = nn.MaxUnpool2d(2, 2)
        self.deconv2_1 = self._deconv_unit(128, 128)
        self.deconv2_2 = self._deconv_unit(128, 64)
        self.unpool1 = nn.MaxUnpool2d(2, 2)
        self.deconv1_1 = self._deconv_unit(64, 64)
        self.deconv1_2 = self._deconv_unit(64, 3)
        self.conv0 = self._conv_unit(3, 1)

        self._init_weights()

    def forward(self, left_images):

        encode = self.conv1_1(left_images)
        encode = self.conv1_2(encode)

        encode, idx1 = self.pool1(encode)

        encode = self.conv2_1(encode)
        encode = self.conv2_2(encode)

        encode, idx2 = self.pool2(encode)

        encode = self.conv3_1(encode)
        encode = self.conv3_2(encode)
        encode = self.conv3_3(encode)

        encode, idx3 = self.pool3(encode)

        encode = self.conv4_1(encode)
        encode = self.conv4_2(encode)
        encode = self.conv4_3(encode)

        encode, idx4 = self.pool4(encode)

        encode = self.conv5_1(encode)
        encode = self.conv5_2(encode)
        encode = self.conv5_3(encode)

        encode, idx5 = self.pool5(encode)

        encode = self.conv6(encode)

        # ==================== DECODER ==============

        decode = self.unpool5(encode, idx5)

        decode = self.deconv5_1(decode)
        decode = self.deconv5_2(decode)
        decode = self.deconv5_3(decode)

        decode = self.unpool4(decode, idx4)

        decode = self.deconv4_1(decode)
        decode = self.deconv4_2(decode)
        decode = self.deconv4_3(decode)

        decode = self.unpool3(decode, idx3)

        decode = self.deconv3_1(decode)
        decode = self.deconv3_2(decode)
        decode = self.deconv3_3(decode)

        decode = self.unpool2(decode, idx2)

        decode = self.deconv2_1(decode)
        decode = self.deconv2_2(decode)

        decode = self.unpool1(decode, idx1)

        decode = self.deconv1_1(decode)
        decode = self.deconv1_2(decode)
        decode = self.conv0(decode)

        return decode

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _conv_unit(self, input_ch, out_ch):
        conv_unit = nn.Sequential(
            nn.Conv2d(input_ch, out_ch, kernel_size=(3, 3), padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        return conv_unit

    def _deconv_unit(self, input_ch, out_ch):
        deconv_unit = nn.Sequential(
            nn.ConvTranspose2d(input_ch, out_ch, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU()
        )

        return deconv_unit
