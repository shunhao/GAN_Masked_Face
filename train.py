from torch.utils.data import DataLoader

from models import GAN_256, Unet_256
from dataset import FacadeDataset_504
from utils import visual_result_singleModel

def train(num_epoch, batch_size, learning_rate, l1_weight):
    train_data = FacadeDataset_504(flag='train', data_range=(0, 3))
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_data = FacadeDataset_504(flag='train', data_range=(0, 1))
    val_loader = DataLoader(val_data, batch_size=batch_size)
    visual_data = next(iter(DataLoader(val_data, batch_size=3)))

    GAN = GAN_256(learning_rate=learning_rate, l1_weight=l1_weight)

    start_epoch = 0
    if GAN.trained_epoch > 0:
        start_epoch = GAN.trained_epoch + 1

    for epoch in range(start_epoch, num_epoch):
        if epoch % 50 == 0:
            print('----------------------------- epoch {:d} -----------------------------'.format(epoch + 1))
        GAN.train_one_epoch(train_loader, val_loader, epoch)
        
        if epoch % 200 == 0:
            visual_result_singleModel(visual_data[0], visual_data[1], GAN.G_model, GAN.trained_epoch+1)

    GAN.plot_loss()

    GAN.save()


if __name__ == '__main__':
    train(num_epoch=5001, batch_size=1, learning_rate=1e-4, l1_weight=5)
