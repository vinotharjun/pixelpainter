from painter import *


class UNetTrainer():
    def __init__(self,
                 generator,
                 train_loader,
                 validation_loader,
                 loss,
                 load: bool = False):
        self.generator = generator
        self.device = device
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.feat_loss = loss.to(device)
        if load == True:
            self.load_checkpoint()


#         self.validation_loader = validation_loader
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=1e-4,
                                            betas=(0., 0.99))

    def save_checkpoint(self, file="checkpoint"):
        torch.save(self.generator.state_dict(),
                   "./storage/saved_models/{}.pt".format(file))

    def load_checkpoint(self, file="checkpoint"):
        self.generator.load_state_dict(
            torch.load("./storage/saved_models/{}.pt".format(file)))

    def train(self, epoch, b=0, eb=-1):

        self.generator.train()
        self.discriminator.train()  # training mode enables batch normalization

        losses_c = AverageMeter()  # content loss
        losses_a = AverageMeter()  # adversarial loss in the generator
        losses_d = AverageMeter()  # adversarial loss in the discriminator

        for i, imgs in enumerate(self.train_loader):
            if i <= b and epoch == eb:
                continue
            lr_imgs = imgs[0].to(device)
            hr_imgs = imgs[1].to(device)

            generated = self.generator(lr_imgs)
            content_loss = self.feat_loss(generated, hr_imgs)

            score_real = self.discriminator(hr_imgs)
            score_fake = self.discriminator(generated)
            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()
            adversarial_loss_rf = self.adv_loss(
                discriminator_rf, torch.zeros_like(discriminator_rf))
            adversarial_loss_fr = self.adv_loss(
                discriminator_fr, torch.ones_like(discriminator_fr))
            adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

            perceptual_loss = content_loss + self.beta * adversarial_loss
            self.optimizer_G.zero_grad()
            perceptual_loss.backward()
            self.optimizer_G.step()
            losses_c.update(content_loss.item(), lr_imgs.size(0))
            losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

            # DISCRIMINATOR UPDATE

            # Discriminate super-resolution (SR) and high-resolution (HR) images
            score_real = self.discriminator(hr_imgs)
            score_fake = self.discriminator(generated.detach())
            discriminator_rf = score_real - score_fake.mean()
            discriminator_fr = score_fake - score_real.mean()
            adversarial_loss_rf = self.adv_loss(
                discriminator_rf, torch.ones_like(discriminator_rf))
            adversarial_loss_fr = self.adv_loss(
                discriminator_fr, torch.zeros_like(discriminator_fr))
            adversarial_loss = (adversarial_loss_fr + adversarial_loss_rf) / 2

            # Back-prop.
            self.optimizer_D.zero_grad()
            adversarial_loss.backward()
            self.optimizer_D.step()
            losses_d.update(adversarial_loss.item(), hr_imgs.size(0))
            if i % 100 == 0:
                with torch.no_grad():
                    self.save_checkpoint()
                    self.generator.eval()
                    preds = self.generator(d[0].to(device))
                    print(
                        "Epoch:{} [{}/{}] content loss :{} advloss:{} discLoss:{}"
                        .format(epoch, i, len(dataloader), losses_c.avg,
                                losses_a.avg, losses_d.avg))
                    losses_c.reset()
                    losses_a.reset()
                    losses_d.reset()
                    # img_grid = torch.cat((imgs_lr, preds.cpu().detach(),d["hr"]), -1)
                    # wandb.log({"images":save_result(img_grid)})

                    wandb.log({
                        "input":
                        save_result(d[0], denormalize=True),
                        "output":
                        save_result(preds, denormalize=True),
                        "ground_truth":
                        save_result(d[1], denormalize=True)
                    })
                    self.generator.train()
            del lr_imgs, hr_imgs, generated, score_real, score_fake

    def train_model(self, start=0, end=100, b=0, eb=-1):
        for epoch in range(start, end):
            if epoch == start:
                b = b
            else:
                b = 0
            self.train(epoch=epoch, b=b, eb=eb)
