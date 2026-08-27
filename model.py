"""
ในการจัดจำแนกเพื่อตรวจจับจุดภาพ (pixel) ที่เป็นเมฆในตัวอย่างภาพถ่ายดาวเทียม Landsat ออกจากภาพพื้นหลัง (Background) ผมได้พิจารณาเลือกใช้ U-net
ที่ผมเลือกใช้ U-Net เพราะงานนี้ต้องการทั้งบริบทเชิงพื้นที่ที่กว้างและความแม่นยำเชิงตำแหน่ง (แยกเมฆขนาดใหญ่ออกจากพื้นหลัง และมีขอบเขตของเมฆที่คมชัด) 
ดังนั้น skip connection ใน U-Net จึงช่วยทำหน้าที่ส่งรายละเอียดเชิงพื้นที่ระดับ encoder ไปยัง decoder โดยตรง ซึ่งโครงสร้าง encoder-decoder โดยทั่วไป
ที่ไม่มี skip connection จะสูญเสียรายละเอียดสองส่วนนี้ไป

โดยในการออกแบบนี้ผมใช้ downsampling 3 ระดับ กับ base_channels=32 (เพื่อให้พอดีกับข้อจำกัด กับคอมพิวเตอร์ผมและเวลาเทรนที่มีจำกัด และในขั้นตอนการเตรียมข้อมูล
ที่อยู่ภายในตัวโมเดลเอง มีการสร้าง NDVI Layer จาก input 4-channel ดิบ แล้วต่อเป็นแชนเนลที่ 5 ก่อนเข้า convolution ชั้นแรก การทำแบบนี้เป็นเพิ่มตัวแปรเข้าไปเพื่อให้
สามารถจำแนกเมฆกับท้องฟ้าโปร่งได้ดียิ่งขึ้นแต่ยังคงไม่เปลี่ยน interface ภายนอกของโมเดล

"""

import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.maxpool_conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        x = torch.cat([x1, x2], dim=1)
        return self.conv(x)

class NDVILayer(nn.Module):
    def __init__(self, red_idx=0, nir_idx=3, eps=1e-6):
        super().__init__()
        self.red_idx = red_idx
        self.nir_idx = nir_idx
        self.eps = eps

    def forward(self, x):
        red = x[:, self.red_idx:self.red_idx+1, :, :]
        nir = x[:, self.nir_idx:self.nir_idx+1, :, :]
        ndvi = (nir - red) / (nir + red + self.eps)
        ndvi = torch.clamp(ndvi, -1.0, 1.0)
        return torch.cat([x, ndvi], dim=1)

class UNet(nn.Module):
    def __init__(self, in_channels=4, out_channels=1, base_channels=32):
        super().__init__()
        self.index_layer = NDVILayer()

        self.inc = DoubleConv(in_channels + 1, base_channels)   # +1 เพิ่ม NDVI

        self.down1 = Down(base_channels, base_channels*2)
        self.down2 = Down(base_channels*2, base_channels*4)
        self.down3 = Down(base_channels*4, base_channels*8)
        self.up1 = Up(base_channels*8 + base_channels*4, base_channels*4)
        self.up2 = Up(base_channels*4 + base_channels*2, base_channels*2)
        self.up3 = Up(base_channels*2 + base_channels, base_channels)
        self.outc = nn.Conv2d(base_channels, out_channels, kernel_size=1)

    def forward(self, x):
        x = self.index_layer(x)

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.outc(x)
        return torch.sigmoid(x)

if __name__ == "__main__":
    model = UNet(in_channels=4, out_channels=1, base_channels=32)
    test_input = torch.randn(2, 4, 384, 384)
    output = model(test_input)
 
    print("Input shape:", test_input.shape)
    print("Output shape:", output.shape)
    print("Output range:", output.min().item(), "-", output.max().item())
 
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameter: {total_params:,}")

