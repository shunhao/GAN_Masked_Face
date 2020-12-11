from torch.utils.data import DataLoader

from models import GAN_256
from dataset import FacadeDataset_504
from utils import visual_result_singleModel
import matplotlib.pyplot as plt


def train(num_epoch, batch_size, learning_rate, l1_weight, pretrain_epoch = 0):
    train_data = FacadeDataset_504(flag='train', data_range=(0, 24))
    train_loader = DataLoader(train_data, batch_size=batch_size)
    val_data = FacadeDataset_504(flag='train', data_range=(5, 8))
    val_loader = DataLoader(val_data, batch_size=batch_size)
    visual_data = next(iter(DataLoader(val_data, batch_size=4)))

    GAN = GAN_256(learning_rate=learning_rate, l1_weight=l1_weight)

    start_epoch = 0
    if GAN.trained_epoch > 0:
        start_epoch = GAN.trained_epoch + 1
    lossesD, lossesG_GAN, lossesG_L1 = [], [], []
    for epoch in range(start_epoch, num_epoch + pretrain_epoch):
        if epoch % 50 == 0:
            print('----------------------------- epoch {:d} -----------------------------'.format(epoch + 1))
        if epoch >= pretrain_epoch:
            lossD, lossG_GAN, lossG_L1 = GAN.train_one_epoch(train_loader, val_loader, epoch, pretrain = False)
        else:
            lossD, lossG_GAN, lossG_L1 = GAN.train_one_epoch(train_loader, val_loader, epoch, pretrain = True)

        lossesD.append(lossD)
        lossesG_GAN.append(lossG_GAN)
        lossesG_L1.append(lossG_L1)
        
        if epoch % 50 == 0:
            visual_result_singleModel(visual_data[0], visual_data[1], GAN.G_model, GAN.trained_epoch+1)
            plt.figure()
            plt.plot(lossesD)
            plt.xlabel('epoch')
            plt.title('Dloss')
            plt.savefig('/content/Dloss_history.png')
            plt.close()

            plt.figure()
            plt.plot(lossesG_GAN)
            plt.xlabel('epoch')
            plt.title('G_GAN loss')
            plt.savefig('/content/G_GAN_loss_history.png')
            plt.close()

            plt.figure()
            plt.plot(lossesG_L1)
            plt.xlabel('epoch')
            plt.title('G_L1 loss')
            plt.savefig('/content/G_L1_loss_history.png')
            plt.close()
    #GAN.plot_loss()
    

    GAN.save()


if __name__ == '__main__':
    train(num_epoch=1201, batch_size=5, learning_rate=1e-4, l1_weight=5, pretrain_epoch = 1001)
