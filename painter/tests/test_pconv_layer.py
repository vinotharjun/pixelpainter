from ..networks.layers import *


class LayerTestModel(nn.Module):
    def __init__(self):
        super(LayerTestModel, self).__init__()
        self.layer1 = PartialConv(3, 3, kernel_size=(3, 3), stride=1)
        self.layer2 = PartialConv(3, 8, kernel_size=(3, 3), stride=1)
        self.layer3 = PartialConv(8, 16, kernel_size=(3, 3), stride=1)
        self.layer4 = PartialConv(16, 32, kernel_size=(3, 3), stride=1)
        self.layer5 = PartialConv(32, 32, kernel_size=(3, 3), stride=1)
        self.layer6 = PartialConv(32, 32, kernel_size=(3, 3), stride=1)
        self.layer7 = PartialConv(32, 32, kernel_size=(3, 3), stride=1)
        self.layer8 = PartialConv(32, 32, kernel_size=(3, 3), stride=1)

    def forward(self, image, mask):
        image, mask1 = self.layer1(image, mask)
        image, mask2 = self.layer2(image, mask1)
        image, mask3 = self.layer3(image, mask2)
        image, mask4 = self.layer4(image, mask3)
        image, mask5 = self.layer5(image, mask4)
        image, mask6 = self.layer6(image, mask5)
        image, mask7 = self.layer7(image, mask6)
        image, mask8 = self.layer8(image, mask7)
        return image, mask1, mask2, mask3, mask4, mask5, mask6, mask7, mask8
