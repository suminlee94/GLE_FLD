"""
@author : Sumin Lee
A Global-local Embedding Module for Fashion Landmark Detection
ICCV 2019 Workshop 'Computer Vision for Fashion, Art, and Design'
"""
import os
from arg import argument_parser
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from models import Network
from utils import cal_loss, Evaluator
import utils

parser = argument_parser()
args = parser.parse_args()

def main():
    # random seed
    seed = 1234
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # load dataset
    if args.dataset[0] == 'deepfashion':
        ds = pd.read_csv('./Anno/df_info.csv')
        from dataset import DeepFashionDataset as DataManager
    elif args.dataset[0] == 'fld':
        ds = pd.read_csv('./Anno/fld_info.csv')
        from dataset import FLDDataset as DataManager
    else :
        raise ValueError

    print('dataset : %s' % (args.dataset[0]))
    if not args.evaluate:
        train_dm = DataManager(ds[ds['evaluation_status'] == 'train'], root=args.root)
        train_dl = DataLoader(train_dm, batch_size=args.batchsize, shuffle=True)

        if os.path.exists('models') is False:
            os.makedirs('models')

    test_dm = DataManager(ds[ds['evaluation_status'] == 'test'], root=args.root)
    test_dl = DataLoader(test_dm, batch_size=args.batchsize, shuffle=False)

    # Load model
    print("Load the model...")
    net = Network(dataset=args.dataset, flag=args.glem).cuda()
    if not args.weight_file == None:
        weights = torch.load(args.weight_file)
        if args.update_weight:
            weights = utils.load_weight(net, weights)
        net.load_state_dict(weights)

    # evaluate only
    if args.evaluate:
        print("Evaluation only")
        test(net, test_dl, 0)
        return

    # learning parameters
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.1)

    print('Start training')
    for epoch in range(args.epoch):
        lr_scheduler.step()
        train(net, optimizer, train_dl, epoch)
        test(net, test_dl, epoch)


def train(net, optimizer, trainloader, epoch):
    train_step = len(trainloader)
    net.train()
    for i, sample in enumerate(trainloader):
        for key in sample:
            sample[key] = sample[key].cuda()
        output = net(sample)
        loss = cal_loss(sample, output)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (i + 1) % 10 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'
                  .format(epoch + 1, args.epoch, i + 1, train_step, loss.item()))

    save_file = 'model_%02d.pkl'
    print('Saving Model : ' + save_file % (epoch + 1))
    torch.save(net.state_dict(), './models/'+ save_file % (epoch + 1))


def test(net, test_loader, epoch):
    net.eval()
    test_step = len(test_loader)
    print('\nEvaluating...')
    with torch.no_grad():
        evaluator = Evaluator()
        for i, sample in enumerate(test_loader):
            for key in sample:
                sample[key] = sample[key].cuda()
            output = net(sample)
            evaluator.add(output, sample)
            if (i + 1) % 100 == 0:
                print('Val Step [{}/{}]'.format(j + 1, test_step))

            results = evaluator.evaluate()
            print('Epoch {}/{}'.format(epoch + 1, args.epoch))
            print('|  L.Collar  |  R.Collar  |  L.Sleeve  |  R.Sleeve  |   L.Waist  |   R.Waist  |    L.Hem   |   R.Hem    |     ALL    |')
            print('|   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |   {:.5f}  |'
                  .format(results['lm_dist'][0], results['lm_dist'][1], results['lm_dist'][2], results['lm_dist'][3],
                          results['lm_dist'][4], results['lm_dist'][5], results['lm_dist'][6], results['lm_dist'][7],
                          results['lm_dist_all']))


if __name__ == '__main__':
    main()