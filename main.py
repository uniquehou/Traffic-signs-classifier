import torch
from torch import optim
from torchvision.transforms import ToPILImage
import argparse
from Data import loadData
from Network import AlexNet, Net, PlusNet, StrongNet
from copy import deepcopy
import matplotlib.pyplot as plt
import os
import re

def train(model, data_loader, criterion, optimizer, epochs, args, device, start_epoch=1, best_acc=0):
    fig = plt.figure()
    ax_color = ('#337ab7', '#5cb85c', '#d9534f')    # 训练集、测试集和最佳模型颜色
    ax_loss, ax_acc = fig.add_subplot(1, 2, 1), fig.add_subplot(1, 2, 2)
    ax_loss.set_title('Loss')
    ax_acc.set_title('Accuracy')
    ax_acc.set_ylim(0, 1)    # 准确率的范围为0-1
    train_loss, test_loss = [], []
    train_acc, test_acc = [], []
    best_acc = best_accf
    best_wts = dict()
    plt.ion()
    for epoch in range(start_epoch, epochs+1):
        for phase_id, phase in enumerate(('train', 'test')):
            epoch_loss = 0
            if phase=='train':
                model.train()
                loss_list = train_loss
                acc_list = train_acc
            else:
                model.eval()
                loss_list = test_loss
                acc_list = test_acc
            loader = data_loader[phase]
            acc_cnt = 0
            for batch in loader:
                imgs = batch['image'].to(device)
                label_class = batch['class'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase=='train'):
                    x_class = model(imgs)

                    _loss = criterion[phase_id](x_class, label_class) * len(imgs)
                    _, pre_class = torch.max(x_class, 1)
                    acc_cnt += torch.sum(pre_class==label_class)
                    # loss
                    loss = criterion[phase_id](x_class, label_class)
                    # fotal loss
                    positive_sample = pre_class == label_class
                    negative_sample = pre_class != label_class
                    positive_loss, negative_loss = 0, 0
                    if torch.sum(positive_sample)>0:
                        positive_loss = criterion[phase_id](x_class[positive_sample], label_class[positive_sample]) * args.positive_weight
                    if torch.sum(negative_sample)>0:
                        negative_loss = criterion[phase_id](x_class[negative_sample], label_class[negative_sample]) * args.negative_weight
                    loss = positive_loss + negative_loss
                    epoch_loss += loss.item()

                    if phase=='train':
                        loss.backward()
                        optimizer.step()

            epoch_loss /= len(loader.dataset)
            acc = acc_cnt.float() / len(loader.dataset) * 100.
            # 绘图
            plt.cla()
            loss_list.append(epoch_loss)
            acc_list.append(acc)
            ax_acc.plot(train_acc, label='train', c=ax_color[phase_id])
            ax_acc.plot(test_acc, label='test', c=ax_color[phase_id])
            ax_loss.plot(train_loss, label='train', c=ax_color[phase_id])
            ax_loss.plot(test_loss, label='test', c=ax_color[phase_id])
            if epoch==start_epoch:
                ax_acc.legend(loc='upper left')
                ax_loss.legend(loc='upper right')
            print(f"Epoch: {epoch}, {phase} loss: {epoch_loss}, acc: {acc:.2f}%")

            if phase=='test' and acc>best_acc:
                best_acc = acc
                plt.text(epoch, acc+0.01, '{acc:.2f}%', c=ax_color[2])    # 标注当前最佳模型
                best_wts = deepcopy(model.state_dict())
                torch.save(best_wts, args.best_wts_name)
                print(f"Best acc is {best_acc}, saved model")
            elif phase == 'train' and epoch % 10 == 0:
                torch.save(model.state_dict(), os.path.join(args.model_dir, f'model_{epoch}.pt'))
                print(f"saved model model_{epoch}.pt")
        print("==================================================")
    return best_acc


def finetune(model, data_loader, criterion, optimizer, epochs, args, device):
    best_acc = 0
    best_wts = dict()
    # best model
    if os.path.exists(args.best_wts_name):
        try:
            model.load_state_dict(torch.load(args.best_wts_name))
        except RuntimeError:
            model.load_state_dict(torch.load(args.best_wts_name, map_location=lambda storage, loc: storage))

        loader = data_loader['test']
        model.eval()
        acc = 0
        with torch.no_grad():
            acc_cnt = 0
            for idx, batch in enumerate(loader):
                imgs = batch['image'].to(device)
                label_class = batch['class'].to(device)
                x_class = model(imgs)
                loss = criterion[1](x_class, label_class) * len(imgs)
                _, pred_class = torch.max(x_class, 1)
                acc_cnt += torch.sum(label_class==pred_class)
        acc = acc_cnt.float() / len(loader.dataset) * 100.
        print("Best accuracy is ", acc)
        print("Load best wts")
        best_acc = acc
        best_wts = deepcopy( model.state_dict())

    # last model
    model_name = \
    sorted(os.listdir(args.model_dir), key=lambda x: int(re.match('model_([\d]+).pt', x).group(1)))[-1]
    start_epoch = int(re.match('model_([\d]+).pt', model_name).group(1))
    model.load_state_dict(torch.load(os.path.join(args.model_dir, model_name)))
    print(f"Load lsat model: {model_name}, continue train from {start_epoch} to {epochs}")

    best_acc = train(model, data_loader, criterion, optimizer, epochs, args, device, start_epoch, best_acc)
    return best_acc


def test(model:torch.nn.Module, data_loader, criterion, device, args):
    try:
        model.load_state_dict(torch.load(args.best_wts_name))
    except RuntimeError:
        model.load_state_dict(torch.load(args.best_wts_name, map_location=lambda storage, loc: storage))

    loader = data_loader['test']
    model.eval()
    acc = 0
    total_loss = 0
    with torch.no_grad():
        acc_cnt = 0
        for idx, batch in enumerate(loader):
            imgs = batch['image'].to(device)
            label_class = batch['class'].to(device)
            x_class = model(imgs)
            loss = criterion[1](x_class, label_class) * len(imgs)
            total_loss += loss.item()
            _, pred_class = torch.max(x_class, 1)
            acc_cnt += torch.sum(label_class==pred_class)
            print(f"batch_id: {idx}, loss: {loss.item()}, acc_cnt = {torch.sum(label_class==pred_class)}")
    total_loss /= len(loader.dataset)
    acc = acc_cnt.float() / len(loader.dataset) * 100.
    return total_loss, acc


def predict(model, data_loader, criterion, device, args):
    try:
        model.load_state_dict(torch.load(args.best_wts_name))
    except RuntimeError:
        model.load_state_dict(torch.load(args.best_wts_name, map_location=lambda storage, loc: storage))

    loader = data_loader['test']
    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(loader):
            imgs = batch['image'].to(device)
            label_class = batch['class'].to(device)
            x_class = model(imgs)
            _, pred_class = torch.max(x_class, 1)

            show_cnt = 25  # 随机展示25张
            import numpy as np
            for idx, i in enumerate(np.random.choice(len(imgs), show_cnt, replace=False)):
                plt.subplot(5, 5, idx+1)
                img = ToPILImage()(imgs[i])
                plt.title(f"{pred_class[i]} - {label_class[i]}")
                plt.imshow(img)
            plt.show()
            break


def statictic(model, data_loader, criterion, device, args):
    import numpy as np
    try:
        model.load_state_dict(torch.load(args.best_wts_name))
    except RuntimeError:
        model.load_state_dict(torch.load(args.best_wts_name, map_location=lambda storage, loc: storage))

    loader = data_loader['test']
    bad_dataset = []
    model.eval()
    with torch.no_grad():
        bad_cnt = 0
        for idx, batch in enumerate(loader):
            imgs = batch['image'].to(device)
            label_class = batch['class'].to(device)
            x_class = model(imgs)
            _, pred_class = torch.max(x_class, 1)
            bad_cnt += torch.sum(label_class!=pred_class)
            bad_sample = [label_class!=pred_class]
            bad_dataset.extend([{'origin_size': x, 'class': y} for (x, y) in zip(batch['origin_size'][bad_sample], label_class[bad_sample])])

    test_class_num = np.array([0]*62)
    for sample in loader.dataset:
        test_class_num[sample['class']] += 1
    bad_size = np.array([sample['origin_size'].numpy() for sample in bad_dataset])
    bad_class = np.array([sample['class'] for sample in bad_dataset])
    bad_size_cnt = np.sum(np.all(bad_size < (227, 227), 1)==True)
    bad_class_cnt = test_class_num[np.unique(bad_class)]
    return bad_cnt, bad_dataset


def main(Model:torch.nn.Module, config:dict):
    parse = argparse.ArgumentParser(description='Train')
    parse.add_argument('--train-size', type=int, default=config['train_size'])    # default 128
    parse.add_argument('--train-batch-size', type=int, default=config['train_batch_size'])    # default 128
    parse.add_argument('--test-batch-size', type=int, default=config['test_batch_size'])    # default 30
    parse.add_argument('--epochs', type=int, default=config['epochs'])    # default 100
    parse.add_argument('--lr', type=float, default=config['lr'])    # default 0.001
    parse.add_argument('--momentum', type=float, default=config['momentum'])    # default 0.9
    parse.add_argument('--phase', type=str, default=config['phase'])    # default train
    parse.add_argument('--model-dir', type=str, default=config['model_dir'])    # default traned_model
    parse.add_argument('--positive-weight', type=int, default=config['positive_weight'])
    parse.add_argument('--negative-weight', type=int, default=config['negative_weight'])
    parse.add_argument('--best_wts-name', type=str, default=config['best_wts_name'])
    args = parse.parse_args()
    data_loader = loadData(args)
    print("===> Load Data")

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    model = Model.to(device)

    # train_class_num, test_class_num = torch.FloatTensor([0]*62), torch.FloatTensor([0]*62)
    # for sample in data_loader['train'].dataset:
    #     train_class_num[sample['class']] += 1
    # for sample in data_loader['test'].dataset:
    #     test_class_num[sample['class']] += 1
    # train_loss_weight = torch.mean(train_class_num) / train_class_num
    # test_loss_weight = torch.mean(test_class_num) / test_class_num
    # criterion = [torch.nn.CrossEntropyLoss(weight=train_loss_weight.to(device)), torch.nn.CrossEntropyLoss(weight=test_loss_weight.to(device))]
    criterion = [torch.nn.CrossEntropyLoss(), torch.nn.CrossEntropyLoss()]
    # criterion = torch.nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    # optimizer = optim.RMSprop(model.parameters(), lr=args.lr)
    print("===> Build model")
    if args.phase.capitalize() == 'Train':
        print('===> Start Train')
        acc = train(model, data_loader, criterion, optimizer, args.epochs, args, device)
        print(f"Train finished, best accuracy is {acc}")
    elif args.phase.capitalize() == 'Test':
        print('=== Start test')
        loss, acc = test(model, data_loader, criterion, device, args)
        print(f"Loss: {loss}  Accuracy: {acc}")
    elif args.phase.capitalize() == 'Finetune':
        print('===> Start finetune')
        finetune(model, data_loader, criterion, optimizer, args.epochs, args, device)
    elif args.phase.capitalize() == 'Predict':
        print('===> Start predict')
        predict(model, data_loader, criterion, device, args)
    elif args.phase.capitalize() == 'Statistic':
        bad_cnt, bad_dataset =statictic(model, data_loader, criterion, device, args)


if __name__ == '__main__':
    # AlexNet
    # net = AlexNet()
    # Net
    # net = Net()
    # PlusNet
    net = PlusNet()
    config = net.getConfig()

    # Test
    # Config['phase'] = 'test'
    # Statistic
    # Config['phase'] = 'statistic'
    # config.best_wts_name = 'best_net_wts.pt'
    # Config['train_size'] = 128
    # predict
    config['phase'] = 'predict'
    main(net, config)