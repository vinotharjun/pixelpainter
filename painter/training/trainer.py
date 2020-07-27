from painter import *
from painter.utils import *
import datetime

class Trainer():
    def __init__(self,
                 generator: nn.Module,
                 train_loader: Iterable,
                 validation_loader: Iterable,
                 criterion: Callable,
                 load: bool = False,
                 load_folder: Any[str] = None,
                 learning_rate=2e-4):

        self.device = device
        self.generator = generator.to(self.device)
        self.train_loader = train_loader
        self.validation_loader = validation_loader
        self.criterion = criterion.to(device)
        if load == True and load_folder == True:
            self.load_checkpoint(load_file)
        else:
            print(
                "No checkpoint path provided so loading from the starting point"
            )
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(),
                                            lr=learning_rate,
                                            betas=(0.9, 0.99))

    def save_checkpoint(self, save_folder:Any[str]=None):
        if not os.path.exists(save_folder):
            os.mkdir(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        torch.save(self.generator.state_dict(), save_folder/"model.pth")
        torch.save(self.optimizer_G.state.dict(),save_folder/"optimizer.pth")

    def load_checkpoint(self, load_folder):
        self.generator.load_state_dict(torch.load(load_folder))
        self.optimizer_G.load_state_dict(torch.load(load_folder))

    def train(self, save_file, epoch, b=0, eb=-1, interval_verbose=100):

        self.generator.train()

        losses_train = AverageMeter()  # content loss
        losses_valiation = AverageMeter()  # adversarial loss in the generator
        losses_total_train = AverageMeter()
        psnr_train = AverageMeter()
        psnr_validation = AverageMeter()

        for i, imgs in enumerate(self.train_loader):
            if i <= b and epoch == eb:
                continue
            input_image = imgs["input_image"].to(device)
            input_mask = imgs["mask"].to(device)
            target = imgs["ground_truth"].to(device)

            generated, _ = self.generator(input_image, input_mask)
            loss = self.criterion(input_image, input_mask, generated, target)

            losses_total_train.update(loss.item(), target.size(0))
            if i % interval_verbose == 0:
                with torch.no_grad():
                    save_file = ifnone(self.load)
                    self.save_checkpoint()
                    self.generator.eval()
                    sample = getsample(self.validation_loader)
                    preds = self.generator(sample["input"].to(device),sample["mask"].)
                    print(
                        "Epoch:{} [{}/{}] content loss :{} advloss:{} discLoss:{}"
                        .format(epoch, i, len(dataloader), losses_c.avg,
                                losses_a.avg, losses_d.avg))
                    losses_c.reset()
                    losses_a.reset()
                    losses_d.reset()
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


# #train mse
# train_loss_list =[]
# val_loss_list =[]
# print_loss =[]
# psnr_best = 13.445319618697336
# val_loss_best= 1e9
# for epoch in range(1 ,100):
#   for i,imgs in enumerate(dataloader):
#     if i<=4000 and epoch ==1:
#       continue
#     generator.train()
#     # batches_done = epoch *len(dataloader) + i
#     imgs_lr = imgs[0].to(device)
#     imgs_hr = imgs[1].to(device)
#     optimizer_G.zero_grad()
#     gen_hr = generator(imgs_lr)
#     loss_pixel = feat_loss(gen_hr,imgs_hr)
#     loss_pixel.backward()
#     optimizer_G.step()
#     scheduler.step()
#     print_loss.append(loss_pixel.item())
#     train_loss_list.append(loss_pixel.item())
#     # print("Epoch[{}/{}] loss :{}".format(epoch,i,len(dataloader),loss_pixel.item()))
#     wandb.log({"epoch":epoch,'train_batch_loss': loss_pixel.item(),"lr":optimizer_G.param_groups[0]["lr"]})
#     if i % opt.sample_interval ==0:
#       torch.save(generator.state_dict(),"./storage/saved_models/checkpoint.pt")
#       torch.save(optimizer_G.state_dict(),"./storage/saved_models/optim_checkpoint.pt")
#       with torch.no_grad():
#         generator.eval()
#         preds = generator(d[0].to(device))
# #         preds.clamp_(0,1)
#       # "{} and {}".format("string", 1)
#         print("Epoch:{} [{}/{}] loss :{}".format(epoch,i,len(dataloader),sum(print_loss)/len(print_loss)))
# #         imgs_lr = nn.functional.interpolate(d["lr"], scale_factor=4)
# #         img_grid = torch.cat((imgs_lr, preds.cpu().detach(),d["hr"]), -1)
# #         wandb.log({"images":save_result(img_grid)})
#         # save_image(img_grid, "images/training/b%d.png" % batches_done, nrow=1, normalize=False)
#         wandb.log({"input":save_result(d[0],denormalize=True),"output":save_result(preds,denormalize=True),"ground_truth":save_result(d[1],denormalize=True)})
#         generator.train()
#   with torch.no_grad():
#     val_loss_list=[]
#     psnr_list =[]
#     generator.eval()
#     for i,imgs in enumerate(val_dataloader):
#       val_batch_done = epoch *len(val_dataloader) + i
#       imgs_lr = imgs[0].to(device)
#       imgs_hr = imgs[1].to(device)
#       gen_hr = generator(imgs_lr)
#       loss_pixel = feat_loss(gen_hr,imgs_hr)
#       psnr_vals = validate(imgs_hr,gen_hr)
#       psnr_list.append(psnr_vals)
#       val_loss_list.append(loss_pixel.item())
#       # print("validation batch loss :",loss_pixel.item())
#       print("Epoch:{} [{}/{}] loss :{}".format(epoch,i,len(val_dataloader),loss_pixel.item()))
#       # wandb.log({"epoch":epoch,'validation_batch_loss': loss.pixel.item()})

#     train_loss_final = sum(train_loss_list)/len(train_loss_list)
#     val_loss_final  =sum(val_loss_list)/len(val_loss_list)

#     print("Epoch:",epoch)
#     print("Train Loss :",train_loss_final)
#     print("Validation Loss :",val_loss_final)
#     psnr_final = sum(psnr_list)/len(psnr_list)
#     print("PSNR :",psnr_final)
#     wandb.log({
#         "train_loss_final":train_loss_final,
#         "psnr":psnr_final,
#         "val_loss_final":val_loss_final,
#     })
# #     if val_loss_final<val_loss_best:
# #       print("saving best mse based model")
# #       torch.save(generator.state_dict(),"/content/drive/My Drive/saved_models/generator_perceptual.pt")
# #       # torch.save(optimizer_G.state_dict(),"/content/drive/My Drive/saved_models/optim_perceptual.pt")
# #       val_loss_best = val_loss_final
#     if psnr_final>psnr_best:
#       print("saving best psnr models")
#       torch.save(generator.state_dict(),"./storage/saved_models/generator_psnr_perceptual_1-6-2020.pt")
#       # torch.save(optimizer_G.state_dict(),"/content/drive/My Drive/saved_models/optim_psnr.pt")
#       psnr_best = psnr_final