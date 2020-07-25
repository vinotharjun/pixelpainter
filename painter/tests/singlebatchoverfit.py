import sys
sys.path.append("./pixelpainter/")
from painter import *
from painter.networks.partialUnet import PartialConvUNet
from painter.losses import PartialConvFeatureLoss
from painter.data.openimages_dataset import OpenImages
from painter.utils import getsample


def overfit_on_single_batch(batched_data, model, criterion, optimizer):
    model.train()
    mb = progress_bar(range(0, 140))
    for e in mb:
        input_image = batched_data["input"].to(device).float()
        input_mask = batched_data["mask"].to(device).float()
        target = batched_data["ground_truth"].to(device).float()
        optimizer.zero_grad()
        outputs, _ = model(input_image, input_mask)
        loss_outputs = criterion(input_image, input_mask, outputs, target)
        loss_outputs.backward()
        optimizer.step()
        print(f'Finished loop {e}    :     Loss :{loss_outputs.item()}')


if __name__ == "__main__":

    print("loading model")
    # model = PartialConvUNet().to(device)
    model = PartialConvUNet().to(device)
    print("loading loss")
    criterion = PartialConvFeatureLoss().to(device)
    print("loading optimizer")
    optimizer = Adam(model.parameters(), lr=2e-4)
    print("loading dataset")
    dataset = OpenImages((256, 256), datatype="validation")
    print("loading dataloaedr")
    data_loader = DataLoader(dataset,
                             batch_size=8,
                             num_workers=4,
                             pin_memory=True)
    print("getsample")
    batched_data = getsample(data_loader)
    print("start tests")
    overfit_on_single_batch(batched_data, model, criterion, optimizer)
