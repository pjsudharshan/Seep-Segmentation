import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils.data as data
from util import EarlyStopping, save_nets, save_predictions, load_best_weights
from model import UNet
from dataset import DataFolder
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib as mpl
import argparse

np. random.seed(1000)

if __name__ == '__main__':
    
	
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks', \
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('-tb','--batch_size', type=int, default=8,
                        help='Batch size for Training and Testing', dest='batch_size')
    parser.add_argument('-e','--epochs', type=int, default=100,
                        help='Maximum number of epochs for training', dest='epochs')
    parser.add_argument('-l','--lr', type=float, default=0.001,
                        help='Learning rate.', dest='lr')
    parser.add_argument('-p','--patience', type=float, default=10,
                        help='Early stopping patience.', dest='patience')
    parser.add_argument('-d','--min_delta', type=float, default=0.001,
                        help='Minimum loss improvement for each epoch.', dest='min_delta')

    args = parser.parse_args()
    print(args)
    
    
    train_loader = data.DataLoader(
        dataset=DataFolder('new_dataset/train/train_images_256/', 'new_dataset/train/train_masks_256/', 'train'),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2
    )
    
    valid_loader = data.DataLoader(
        dataset=DataFolder('new_dataset/val/train_images_256/', 'new_dataset/val/train_masks_256/', 'validation'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    test_loader = data.DataLoader(
        dataset=DataFolder('new_dataset/test/train_images_256/', 'new_dataset/test/train_masks_256/', 'evaluate'),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2
    )
    
    model = UNet(1, shrink=1).cuda()
    nets = [model]
    params = [{'params': net.parameters()} for net in nets]
    solver = optim.Adam(params, lr=args.lr)
    
    criterion = nn.CrossEntropyLoss()
    es = EarlyStopping(min_delta=args.min_delta, patience=args.patience)
    
    for epoch in tqdm(range(1, args.epochs+1)):
        with tqdm(total=len(train_loader.dataset), desc=f'Epoch {epoch}/{args.epochs}', unit='img', \
                  position=0, leave=True) as pbar:
    
            train_loss = []
            valid_loss = []
        
            for batch_idx, (img, mask, _) in enumerate(train_loader):
        
                solver.zero_grad()
        
                img = img.cuda()
                mask = mask.cuda()
        
                pred = model(img)
                loss = criterion(pred, mask)
                pbar.set_postfix(**{'loss (batch)': loss.item()})
                loss.backward()
                solver.step()
        
                train_loss.append(loss.item())
                pbar.update(img.shape[0])
            with torch.no_grad():
                for batch_idx, (img, mask, _) in enumerate(valid_loader):
        
                    img = img.cuda()
                    mask = mask.cuda()
        
                    pred = model(img)
                    loss = criterion(pred, mask)
        
                    valid_loss.append(loss.item())
        
            print('[EPOCH {}/{}] Train Loss: {:.4f}; Valid Loss: {:.4f}'.format(
                epoch, args.epochs, np.mean(train_loss), np.mean(valid_loss)
            ))
        
            flag, best, bad_epochs = es.step(torch.Tensor([np.mean(valid_loss)]))
            if flag:
                print('Early stopping criterion met')
                break
            else:
                if bad_epochs == 0:
                    save_nets(nets, 'saved_models')
                    print('Saving current best model')
        
                print('Current Valid loss: {:.4f}; Current best: {:.4f}; Bad epochs: {}'.format(
                    np.mean(valid_loss), best.item(), bad_epochs
                ))
    
    print('Training is over. ')
    
    with torch.no_grad():
        test_loss = []
        for batch_idx, (img, mask, img_fns) in enumerate(test_loader):
    
            model = load_best_weights(model, 'saved_models')
    
            img = img.cuda()
            mask = mask.cuda()
    
            pred = model(img)
            loss = criterion(pred, mask)
    
            test_loss.append(loss.item())
            
            pred_mask = torch.argmax(F.softmax(pred, dim=1), dim=1)
            pred_mask = torch.chunk(pred_mask, chunks=args.batch_size, dim=0)
            save_predictions(pred_mask, img_fns, 'test_predictions')
    
            print('[Testing {}/{}] Test Loss: {:.4f}'.format(
                batch_idx+1, len(test_loader), loss.item()
            ))
    
    print('FINAL Test Loss: {:.4f}'.format(np.mean(test_loss)))
    

    fig = plt.figure(figsize=(15,15))
    
    cmap = mpl.colors.ListedColormap(['black','blue','red', 'green', 'brown', 'cyan','yellow','royalblue'])
    cmap.set_over('royalblue')
    cmap.set_under('black')
    bounds = [0,1,2,3,4,5,6,7,8]
    
    
    cmaplist = [cmap(i) for i in range(cmap.N)]
    cmap = mpl.colors.LinearSegmentedColormap.from_list('Custom cmap', cmaplist, cmap.N)
    norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
    
    for idx, pred in enumerate(pred_mask):
        pred = torch.squeeze(pred, dim=0)
        ax = fig.add_subplot(3,3, idx+1)
        img = ax.imshow(pred.cpu().numpy(), interpolation='none', cmap=cmap, norm=norm)
        fig.colorbar(img)
        plt.tight_layout()
        plt.savefig('sav_images/Day0.svg',transparent=True)