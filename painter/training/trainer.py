from painter import *
from painter.utils import *
import datetime
import wandb



class Trainer():
    def __init__(self,
                 generator: nn.Module,
                 train_loader: Iterable,
                 validation_loader: Iterable,
                 criterion: Callable,
                 load: bool = False,
                 load_folder:str = None,
                 normalize=False,
                 learning_rate=2e-4):
        self.normalize = normalize
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

    def save_checkpoint(self, save_folder: str=None,isbest=False,model_filename="model.pth",optimizer_filename="optimizer.pth"):
        if isbest:
            if not os.path.exists("best_model") and save_folder==None:
                os.mkdir("best_model")
                save_folder="best_model"
            model_filename= save_folder+"/"+model_filename
            optimizer_filename = save_folder+"/"+optimizer_filename
        else:
            if save_folder==None:
                save_folder=datetime.datetime.now().strftime('%Y-%m-%d_%H')
                os.mkdir(save_folder)
            model_filename= save_folder+"/"+model_filename
            optimizer_filename = save_folder+"/"+optimizer_filename

        torch.save(self.generator.state_dict(),model_filename)
        torch.save(self.optimizer_G.state_dict(),optimizer_filename)

    def load_checkpoint(self, load_folder):
        self.generator.load_state_dict(torch.load(load_folder))
        self.optimizer_G.load_state_dict(torch.load(load_folder))

    def train(self,epoch, b=0, eb=-1, interval_verbose=100):

        self.generator.train()

        losses_train = AverageMeter() 
        losses_total_train = AverageMeter()
        # psnr_train = AverageMeter()
        sample = getsample(self.validation_loader)

        for i, imgs in enumerate(self.train_loader):
            print(i)
            if i <= b and epoch == eb:
                continue
            input_image = imgs["input"].to(device)
            input_mask = imgs["mask"].to(device)
            target = imgs["ground_truth"].to(device)

            generated, _ = self.generator(input_image, input_mask)
            loss = self.criterion(input_image, input_mask, generated, target)
            # psnr_value = validate(target,generated)
            # psnr_train.update(psnr_value)

            losses_train.update(loss.item(), target.size(0))
            losses_total_train.update(loss.item(), target.size(0))
            if i % interval_verbose == 0:
                with torch.no_grad():
                    self.save_checkpoint()
                    self.generator.eval()
                    preds,_ = self.generator(sample["input"].to(device),sample["mask"].to(device))
                    if self.normalize==False:
                        preds.clamp_(0,1)
                    print(
                        "Epoch:{} [{}/{}] content loss :{}"
                        .format(epoch, i, len(self.train_loader), losses_train.avg))
                    losses_train.reset()
                    wandb.log({
                        "input":
                        save_result(sample["input"], denormalize=self.normalize),
                        "output":
                        save_result(preds, denormalize=self.normalize),
                        "ground_truth":
                        save_result(sample["ground_truth"], denormalize=self.normalize)
                    })
                    self.generator.train()
            del input_image, input_mask, generated,target
        return losses_total_train.avg

    def validate(self):

        self.generator.eval()
        losses_total_validation = AverageMeter()
        psnr_validation = AverageMeter()

        with torch.no_grad():
            for i, imgs in enumerate(self.validation_loader):
                input_image = imgs["input"].to(device)
                input_mask = imgs["mask"].to(device)
                target = imgs["ground_truth"].to(device)
                generated, _ = self.generator(input_image, input_mask)
                loss = self.criterion(input_image, input_mask, generated, target)
                losses_total_validation.update(loss.item(), target.size(0))
                psnr_value = validate(generated,target)
                psnr_validation.update(psnr_value)

        return losses_total_validation.avg,psnr_validation.avg

    def train_model(self, start=0, end=100, b=0,eb=-1,val_loss_best=1e9,interval_verbose=100):
        for epoch in range(start, end):
            if epoch == start:
                b = b
            else:
                b = 0
            train_loss = self.train(epoch=epoch, b=b, eb=eb,interval_verbose=interval_verbose)
            val_loss,psnr_validation = self.validate()
            print("Epoch:",epoch)
            print("Train Loss :",train_loss)
            print("Validation Loss :",val_loss)
            print("PSNR :",psnr__validation)
            wandb.log({
                "train_loss_final":train_loss,
                "psnr":psnr_validation,
                "val_loss_final":val_loss,
            })
            if val_loss < val_loss_best:
                print("saving best content based model")
                self.save_checkpoint(isbest=True)
                val_loss_best = val_loss



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