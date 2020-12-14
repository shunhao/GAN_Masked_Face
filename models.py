import torch
from torch import optim, nn
from torch.autograd import Variable
import os
import time
from tqdm import tqdm

from network import  Generator_256, Discriminator_256
from utils import *


class GAN_256():
    def __init__(self, learning_rate=1e-4, l1_weight=5, save_path="trained_256_model.pth.tar"):
        self.trained_epoch = 0
        self.learning_rate = learning_rate
        self.best_acc2 = 0.0
        self.best_acc5 = 0.0

        self.device = torch.device('cpu')
        if torch.cuda.is_available():
            self.device = torch.device('cuda:0')

        self.G_model = Generator_256().to(self.device)
        self.D_model = Discriminator_256().to(self.device)
        #try:
        #    self.G_model = Generator_256().cuda()
        #    self.D_model = Discriminator_256().cuda()
        #except TypeError:
        #    print("cuda is not available!")

        #self.G_optimizer = optim.Adam(self.G_model.parameters(), lr=learning_rate)
        #self.G_optimizer = optim.SGD(self.G_model.parameters(), lr=learning_rate, momentum = 0.9, nesterov=True)
        self.G_optimizer = optim.Adadelta(self.G_model.parameters(), lr = 0.1)
        self.D_optimizer = optim.Adam(self.D_model.parameters(), lr=learning_rate)

        self.criterion = nn.BCEWithLogitsLoss(reduction='mean')
        self.L1 = nn.L1Loss()
        self.L2 = nn.MSELoss()
        self.l1_weight = l1_weight

        self.loss_history = []
        self.acc_history = []

        # read trained model
        self.save_path = save_path
        if os.path.exists(self.save_path):
            checkpoint = torch.load(self.save_path)
            self.G_model.load_state_dict(checkpoint['G_state_dict'])
            self.D_model.load_state_dict(checkpoint['D_state_dict'])

            self.G_optimizer.load_state_dict(checkpoint['G_optimizer_state_dict'])
            self.D_optimizer.load_state_dict(checkpoint['D_optimizer_state_dict'])

            self.trained_epoch = checkpoint['trained_epoch']
            self.loss_history = checkpoint['loss_history']
            self.acc_history = checkpoint['acc_history']
            self.best_acc2 = checkpoint['best_acc2']
            self.best_acc5 = checkpoint['best_acc5']

    def train_one_epoch(self, train_loader, val_loader, epoch, pretrain):
        self.trained_epoch = epoch
        if epoch % 50 == 0:
          self.l1_weight = self.l1_weight * 0.95
        start = time.time()
        #lossesD, lossesD_real, lossesD_fake, lossesG, lossesG_GAN, \
        #lossesG_L1, Dreals, Dfakes = [], [], [], [], [], [], [], []

        lossesD, lossesG_GAN, lossesG_L1 = 0, 0, 0
        self.G_model.train()
        self.D_model.train()

        # for gray, color in tqdm(self.train_loader):
        for gray, img_ab in tqdm(train_loader):

            #lossD = D_loss_real + D_loss_fake
            #gray1 = gray.clone().to(self.device)
            #img_ab1 = img_ab.clone().to(self.device)
            #gray = Variable(gray.to(self.device).detach())
            #img_ab = Variable(img_ab.to(self.device).detach())

            # train D with real mask image
            #if not pretrain:
            self.D_model.zero_grad()
            label = torch.FloatTensor(img_ab.size(0)).to(self.device)

            D_output_real = self.D_model(img_ab.detach().to(self.device))
            label_real = Variable(label.fill_(1))
            
            D_loss_real = self.criterion(D_output_real.reshape(img_ab.size(0)), label_real)

            D_loss_real.backward()
            lossesD = D_loss_real.item()
            
            # train D with no mask
            if pretrain:
                label_fake = Variable(label.fill_(0))
                D_output_nomask = self.D_model(gray.detach().to(self.device))
                D_loss_nomask = self.criterion(D_output_nomask.reshape(gray.size(0)), label_fake)
                D_loss_nomask.backward()
                lossesD += D_loss_nomask.item()

            # train D with Generator
            #if not pretrain:
    
            label_fake = Variable(label.fill_(0))
            fake_img = self.G_model(gray.detach().to(self.device))
            D_output_fake = self.D_model(fake_img)
            D_loss_fake = self.criterion(D_output_fake.reshape(gray.size(0)), label_fake)
            #D_loss = self.L2(D_output_fake.reshape(D_output_fake.shape[0], -1), D_output_real.reshape(D_output_real.shape[0], -1))
            D_loss_fake.backward()
            lossesD += D_loss_fake.item()
          
            self.D_optimizer.step()

            # train G
            self.G_model.zero_grad()

            #
            if not pretrain:
                fake_img_G = self.G_model(gray.detach().to(self.device))
                D_output_fake_G = self.D_model(fake_img_G)
                label = torch.FloatTensor(img_ab.size(0)).to(self.device)
                label_real = Variable(label.fill_(1))
                lossG_GAN = self.criterion(D_output_fake_G.reshape(gray.size(0)), label_real)
                #lossG_GAN = self.L2(D_output_fake_G.view(D_output_fake_G.shape[0], -1), D_output_real_G.view(D_output_real_G.shape[0], -1))

                #fake_img_G[:, :, 50:, 20:-20] *= 0.2
                #gray2[:, :, 50:, 20:-20] *= 0.2
                #
                #lossG_L1 = self.L2(fake_img_G.view(fake_img_G.size(0), -1), gray2.view(gray2.size(0), -1))
            
                #lossG =  lossG_GAN #+ self.l1_weight * lossG_L1

                lossG_GAN.backward()
                lossesG_GAN += lossG_GAN.item() 
                #lossesG_L1 += lossG_L1.item()
            #else:
              #  lossG = self.L2(fake_img.view(fake_img.size(0), -1), gray.view(gray.size(0), -1))
            #lossG.backward()

            # pretrain only
            if pretrain:
                #temptorch = torch.cat([gray2, img_ab2])
                fake_img_nomask = self.G_model(gray.detach().to(self.device))
                #lossG2 = self.L2(fake_img_nomask[:, :, 50:, 20:-20].reshape(fake_img_nomask.size(0), -1), gray2[:, :, 50:, 20:-20].reshape(gray2.size(0), -1))
                lossG2 = self.L2(fake_img_nomask.reshape(fake_img_nomask.size(0), -1), gray.detach().reshape(gray.size(0), -1).to(self.device))
                lossG2.backward()
                lossesG_L1 += lossG2.item()
                #fake_img_nomask = self.G_model(gray.detach().to(self.device))
                #lossG2 = self.L2(fake_img_nomask[:, :, 50:, 20:-20].reshape(fake_img_nomask.size(0), -1), img_ab.detach()[:, :, 50:, 20:-20].reshape(img_ab2.size(0), -1).to(self.device))
                #lossG2.backward()
                #lossesG_L1 += lossG2.item()
                fake_img_mask = self.G_model(img_ab.detach().to(self.device))
                lossG2 = self.L2(fake_img_mask.reshape(fake_img_mask.size(0), -1), img_ab.detach().reshape(img_ab.size(0), -1).to(self.device))
                lossG2.backward()
                lossesG_L1 += lossG2.item()
                
                

            #Dfake = D_output.data.mean()
            self.G_optimizer.step()

            #lossesD.append(lossD)
            #lossesD_real.append(D_loss_real)
            #lossesD_fake.append(D_loss_fake)
            #lossesG.append(lossG)
            #lossesG_GAN.append(lossG_GAN)
            #lossesG_L1.append(lossG_L1)
            #Dreals.append(Dreal)
            #Dfakes.append(Dfake)

        end = time.time()
        #lossD = torch.stack(lossesD).mean().item()
        #D_loss_real = torch.stack(lossesD_real).mean().item()
        #D_loss_fake = torch.stack(lossesD_fake).mean().item()
        #lossG = torch.stack(lossesG).mean().item()
        #lossG_GAN = torch.stack(lossesG_GAN).mean().item()
        #lossG_L1 = torch.stack(lossesG_L1).mean().item()
        #Dreal = torch.stack(Dreals).mean().item()
        #Dfake = torch.stack(Dfakes).mean().item()
        #print('loss_D: %.3f (real: %.3f fake: %.3f)  loss_G: %.3f (GAN: %.3f L1: %.3f) D(real): %.3f  '
        #      'D(fake): %3f  elapsed time %.3f' %
        #      (lossD, D_loss_real, D_loss_fake, lossG, lossG_GAN, lossG_L1, Dreal, Dfake, end-start))

        #lossD_val, lossG_val, acc_2, acc_5 = self.validate(val_loader)
        #if acc_2 > self.best_acc2 and acc_5 > self.best_acc5:
        if epoch % 100 == 0:
             #self.best_acc2 = acc_2
             #self.best_acc5 = acc_5
             self.save()
        return lossesD, lossesG_GAN, lossesG_L1
        # update history
        #loss_all = [lossD, D_loss_real, D_loss_fake, lossG, lossG_GAN, lossG_L1, lossD_val,
        #            lossG_val, Dreal, Dfake]

        #if len(self.loss_history):
        #    self.loss_history = np.hstack((self.loss_history, np.vstack(loss_all)))
        #    self.acc_history = np.hstack((self.acc_history, np.vstack([acc_2, acc_5])))
        #else:
        #    self.loss_history = np.vstack(loss_all)
        #    self.acc_history = np.vstack([acc_2, acc_5])

    def validate(self, val_loader):
        lossesD, lossesG, cnt = 0.0, 0.0, 0
        acc_2_list, acc_5_list = [], []

        with torch.no_grad():
            self.G_model.eval()
            self.D_model.eval()

            for gray, color in tqdm(val_loader):
                gray = Variable(gray.cuda())
                color = Variable(color.cuda())

                # for D_model
                label = torch.FloatTensor(color.size(0)).cuda()
                label_real = Variable(label.fill_(1))
                lossD_real = self.criterion(self.D_model(color).reshape(color.size(0)), label_real)

                fake_img = self.G_model(gray)
                label_fake = Variable(label.fill_(0))
                pred_D_fake = self.D_model(fake_img.detach())
                lossD_fake = self.criterion(pred_D_fake.reshape(gray.size(0)), label_fake)

                lossD = lossD_real.item() + lossD_fake.item()

                # for G_model
                lossG_GAN = self.criterion(pred_D_fake.reshape(gray.size(0)), label_real)
                lossG_L1 = self.L1(fake_img.view(fake_img.size(0), -1), color.view(color.size(0), -1))
                lossG = lossG_GAN.item() + 100 * lossG_L1.item()

                lossesD += lossD
                lossesG += lossG
                cnt += 1

                acc_2, acc_5 = val_accuracy(fake_img, color)
                acc_2_list.append(acc_2)
                acc_5_list.append(acc_5)

            print('loss_D: %.3f loss_G: %.3f' % (lossesD/cnt, lossesG/cnt))

            acc_2 = torch.stack(acc_2_list).mean().item()
            acc_5 = torch.stack(acc_5_list).mean().item()
            print('GAN: 2%% Validation Accuracy = %.4f' % acc_2)
            print('GAN: 5%% Validation Accuracy = %.4f' % acc_5)

            return lossesD/cnt, lossesG/cnt, acc_2, acc_5

    def test(self, test_loader):
        lossesD, lossesG, cnt = 0.0, 0.0, 0
        acc_2_list, acc_5_list = [], []

        with torch.no_grad():
            self.G_model.eval()
            self.D_model.eval()

            for gray, color in tqdm(test_loader):
                gray = Variable(gray.cuda())
                color = Variable(color.cuda())

                # for D_model
                label = torch.FloatTensor(color.size(0)).cuda()
                label_real = Variable(label.fill_(1))
                lossD_real = self.criterion(torch.squeeze(self.D_model(color)), label_real)

                fake_img = self.G_model(gray)
                label_fake = Variable(label.fill_(0))
                pred_D_fake = self.D_model(fake_img.detach())
                lossD_fake = self.criterion(torch.squeeze(pred_D_fake), label_fake)

                lossD = lossD_real.item() + lossD_fake.item()

                # for G_model
                lossG_GAN = self.criterion(torch.squeeze(pred_D_fake), label_real)
                lossG_L1 = self.L1(fake_img.view(fake_img.size(0), -1), color.view(color.size(0), -1))
                lossG = lossG_GAN.item() + 100 * lossG_L1.item()

                lossesD += lossD
                lossesG += lossG
                cnt += 1

                acc_2, acc_5 = val_accuracy(fake_img, color)
                acc_2_list.append(acc_2)
                acc_5_list.append(acc_5)

            print('loss_D: %.3f loss_G: %.3f' % (lossesD / cnt, lossesG / cnt))

            acc_2 = torch.stack(acc_2_list).mean().item()
            acc_5 = torch.stack(acc_5_list).mean().item()
            print('GAN: 2%% test Accuracy = %.4f' % acc_2)
            print('GAN: 5%% test Accuracy = %.4f' % acc_5)

            return lossesD / cnt, lossesG / cnt, acc_2, acc_5

    def save(self):
        torch.save({'G_state_dict': self.G_model.state_dict(),
                    'D_state_dict': self.D_model.state_dict(),
                    'G_optimizer_state_dict': self.G_optimizer.state_dict(),
                    'D_optimizer_state_dict': self.D_optimizer.state_dict(),
                    'loss_history': self.loss_history,
                    'acc_history': self.acc_history,
                    'best_acc2': self.best_acc2,
                    'best_acc5': self.best_acc5,
                    'trained_epoch': self.trained_epoch}, self.save_path)

    def plot_loss(self):
        loss_name = ['lossD_train', 'lossD_fake', 'lossD_real', 'lossG_train', 'lossG_GAN',
                     'lossG_L1', 'lossD_val', 'lossG_val', 'Dreal', 'Dfake']

        if not os.path.exists("loss_plot/"):
            print('creating directory loss_plot/')
            os.mkdir("loss_plot/")

        for i in range(self.loss_history.shape[0]-2):
            plt.figure()
            plt.plot(self.loss_history[i, :])
            plt.xlabel('epoch'), plt.title('{}'.format(loss_name[i]))
            plt.savefig('./loss_plot/{}_history.png'.format(loss_name[i]))
            plt.clf()
        for i in range(self.loss_history.shape[0]-2, self.loss_history.shape[0]):
            plt.figure()
            plt.plot(self.loss_history[i, :])
            plt.xlabel('epoch'), plt.title('{}'.format(loss_name[i]))
            plt.ylim([-0.1, 1.1])
            plt.savefig('./loss_plot/{}_history.png'.format(loss_name[i]))
            plt.clf()

        plt.figure()
        plt.plot(self.acc_history[0], label='2%% accuracy')
        plt.plot(self.acc_history[1], label='5%% accuracy')
        plt.title('Accuracy History GAN'), plt.legend(), plt.ylim([0,1])
        plt.savefig('./loss_plot/accuracy_history_GAN.png')

