from painter import *
from .loss_utils import *


class PartialConvFeatureLoss(nn.Module):
    def __init__(self, layer_wgts=[20, 70, 10]):
        super().__init__()
        self.m_feat = torchvision.models.vgg16_bn(
            True, progress=False).features.to(device).eval()
        for k, v in self.m_feat.named_parameters():
            v.requires_grad = False
        blocks = [i - 1 for i, o in enumerate(children(self.m_feat))]
        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = [SaveFeatures(i) for i in self.loss_features]
        self.wgts = layer_wgts
        self.base_loss = F.l1_loss

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.stored.clone() if clone else o.stored) for o in self.hooks]

    def perceptual_loss(self, composed_feat, out_feat, target_feat):
        loss = 0.0
        for f_out, f_target, f_composed, w in zip(out_feat, target_feat,
                                                  composed_feat, self.wgts):
            loss += self.base_loss(f_out, f_target)
            loss += self.base_loss(f_composed, f_target)
        return loss

    def style_loss(self, composed_feat, out_feat, target_feat):
        loss = 0.0
        for f_out, f_target, f_composed, w in zip(out_feat, target_feat,
                                                  composed_feat, self.wgts):
            loss += gram_loss(f_out, f_target)
            loss += gram_loss(f_composed, f_target)
        return loss

    def total_variation_loss(self, image):
        loss = self.base_loss(image[:, :, :, :-1],
                              image[:, :, :, 1:]) + self.base_loss(
                                  image[:, :, :-1, :], image[:, :, 1:, :])
        return loss

    def forward(self, input_x, mask, output, target, send_details=False):
        composed_output = (input_x * mask) + (output * (1 - mask))
        composed_feat = self._make_features(composed_output, clone=True)
        target_feat = self._make_features(target, clone=True)
        out_feat = self._make_features(output)
        loss_dict = {}
        loss_dict["hole"] = self.base_loss((1 - mask) * output,
                                           (1 - mask) * target) * 6.0
        loss_dict["valid"] = self.base_loss(mask * output, mask * target) * 1.0
        loss_dict["perceptual"] = self.perceptual_loss(composed_feat, out_feat,
                                                       target_feat) * 0.05
        loss_dict["style"] = self.style_loss(composed_feat, out_feat,
                                             target_feat) * 240.0
        loss_dict["regularization"] = self.total_variation_loss(
            composed_output) * 2.0
        return sum(loss_dict.values())

    def __del__(self):
        for i in self.hooks:
            i.remove()
