# coding=utf-8
from __future__ import absolute_import, division, print_function
import argparse
import os
import random
import numpy as np
import time
from sklearn.metrics import confusion_matrix
import torch
import torch.optim.lr_scheduler as lr_scheduler
from tqdm import tqdm
from sklearn import metrics
from models import alg
from models.loss import get_loss_func
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from utils.dataloader import get_loader



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def save_model(args, model):
    model_to_save = model.module if hasattr(model, 'module') else model
    model_checkpoint = os.path.join(args.train_resput_dir, "%s_checkpoint.bin" % args.name)
    torch.save(model_to_save.state_dict(), model_checkpoint)
    print(f"Saved model checkpoint to [DIR: {args.train_resput_dir}]")


def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params / 1000000


def set_seed(args):
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def setup(args):
    # Prepare model
    if args.dataset == "idrid":
        num_classes = 5
        num_classes2 = 3
    elif args.dataset == "kneeoa":
        num_classes = 5
    elif args.dataset == "messidor":
        num_classes = 4

    args.num_classes = num_classes
    args.num_classes2 = num_classes2
    algorithm_class = alg.get_algorithm_class(args.method)
    model = algorithm_class(args).to(args.device)
    num_params = count_parameters(model)
    print(f"model parameter size: {num_params:.2f} M")
    return args, model


def qwk(conf_mat):
    num_ratings = len(conf_mat)
    num_scored_items = float(np.sum(conf_mat))
    hist_rater_a = np.sum(conf_mat, axis=1)
    hist_rater_b = np.sum(conf_mat, axis=0)
    numerator = 0.0
    denominator = 0.0
    for i in range(num_ratings):
        for j in range(num_ratings):
            expected_count = (hist_rater_a[i] * hist_rater_b[j]
                              / num_scored_items)
            d = pow(i - j, 2.0) / pow(num_ratings - 1, 2.0)
            numerator += d * conf_mat[i][j] / num_scored_items
            denominator += d * expected_count / num_scored_items
    return 1.0 - numerator / denominator


def eval_messidor(args, model, test_loader, optimizer, scheduler, current_epoch=-1):
    """Validation"""
    model.eval()
    results = dict()
    start = time.time()
    results['ep'] = current_epoch
    results['lr'] = optimizer.param_groups[0]['lr']
    val_loss = 0
    num_correct = 0
    loss_fct =  torch.nn.CrossEntropyLoss()
    loop = tqdm(enumerate(test_loader), total=len(test_loader), desc='test')
    running_smaple = 0
    label_list = []
    pred_list = []
    y_true = []
    y_pred = []
    y_prob = []
    for step, batch in loop:
        batch = tuple(t.to(args.device) for t in batch)
        x, y = batch
        with torch.no_grad():
            logits = select_first(model(x))
            loss = loss_fct(logits, y)
            val_loss += len(y) * loss.item()
            pred = logits.argmax(dim=1)
            pred = torch.clamp(pred - 1, 0, 1)
            y = torch.clamp(y - 1, 0, 1)
            num_correct += torch.eq(pred, y).sum().float().item()
            running_smaple += len(y)
            label_list += list(y.cpu().detach().numpy())
            pred_list += list(pred.cpu().detach().numpy())
            loop.set_description(f'(Valid) Epoch [{current_epoch:<3d}/{args.epochs}]')
            info = {'loss': '{:.6f}'.format(val_loss/running_smaple), "acc": '{:.4f}'.format(num_correct/running_smaple)}
            loop.set_postfix(info)
            
            y_true += list(y.cpu().detach().numpy())
            y_pred += list(pred.cpu().detach().numpy())
            out1 = torch.softmax(logits, 1)
            out1 = out1.cpu().detach().numpy()
            y_prob.append(out1)
    
    results['ac'] = num_correct / len(test_loader.dataset)
    results['loss'] = val_loss / len(test_loader.dataset)
    results['cm'] = confusion_matrix(label_list, pred_list, labels=list(np.arange(2)))
    results['qwk'] = qwk(results['cm'])
    results['time'] = (time.time() - start ) / 60

    # 4class to 2class
    y_prob = np.concatenate(y_prob, axis=0)
    y_prob_1 = np.sum(y_prob[:,0:2], axis=1, keepdims=True)
    y_prob_2 = np.sum(y_prob[:,2: ], axis=1, keepdims=True)
    y_prob = np.concatenate([y_prob_1, y_prob_2], axis=1)
    f1 = metrics.f1_score(y_true, y_pred)
    p = metrics.precision_score(y_true, y_pred)
    r = metrics.recall_score(y_true, y_pred)
    auc = metrics.roc_auc_score(y_true, y_prob[:,1])
    results['auc'] = auc
    results['pre'] = p
    results['rec'] = r
    results['f1'] = f1
    # print('DR: auc:{:.4f}, ac:{:.4f}, Pre:{:.4f}, Re:{:.4f}, F1:{:.4f}'.format(auc, valid_acc, p, r, f1))
    return results


def select_first(variable):
    if isinstance(variable, (list,tuple)):
        return variable[0]
    return variable

def select_first_two(variable):
    if isinstance(variable, (list,tuple)):
        return variable[0:2]
    return variable

# compute joint ac
def num_consistent(preds1, preds2, labels1, labels2):
    assert len(preds1) == len(preds2) == len(labels1) == len(labels2)
    n = len(preds1)
    corrects = 0
    for i in range(n):
        if preds1[i] == labels1[i] and preds2[i] == labels2[i]:
            corrects += 1
    return corrects


def eval_one_epoch(args, model, val_loader, optimizer, scheduler, current_epoch=-1):
    """Validation"""
    model.eval()
    results = dict()
    start = time.time()
    results['ep'] = current_epoch
    results['lr'] = optimizer.param_groups[0]['lr']
    val_loss = 0
    num_correct = 0
    num_correct2 = 0
    num_correct_joint = 0
    loss_fct =  torch.nn.CrossEntropyLoss()
    loop = tqdm(enumerate(val_loader), total=len(val_loader), desc='val')
    running_smaple = 0
    label_list = []
    pred_list = []
    for step, batch in loop:
        batch = tuple(t.to(args.device) for t in batch)
        x, y, y2 = batch
        with torch.no_grad():
            logits, logits2 = select_first_two(model(x))
            loss = loss_fct(logits, y) + loss_fct(logits2, y2)
            val_loss += len(y) * loss.item()
            pred = logits.argmax(dim=1)
            pred2 = logits2.argmax(dim=1)
            if args.dataset == 'messidor':
                pred = torch.clamp(pred - 1, 0, 1)
                y = torch.clamp(y - 1, 0, 1)
            num_correct += torch.eq(pred, y).sum().float().item()
            num_correct2 += torch.eq(pred2, y2).sum().float().item()
            num_correct_joint += num_consistent(pred, pred2, y.detach(), y2.detach())
            running_smaple += len(y)
            loop.set_description(f'(Valid) Epoch [{current_epoch:<3d}/{args.epochs}]')
            info = {'loss': '{:.6f}'.format(val_loss/running_smaple), "joint acc": '{:.4f}'.format(num_correct_joint/running_smaple)}
            loop.set_postfix(info)

    results['joint_ac'] = num_correct_joint / len(val_loader.dataset)
    results['ac'] = num_correct / len(val_loader.dataset)
    results['ac2'] = num_correct2 / len(val_loader.dataset)
    results['loss'] = val_loss / len(val_loader.dataset)
    results['time'] = (time.time() - start ) / 60
    return results


def train_one_epoch(args, model, train_loader, optimizer, scheduler, current_epoch=-1):
    """ Train the model """
    model.train()
    results = dict()
    results['ep'] = current_epoch
    results['lr'] = optimizer.param_groups[0]['lr']
    start = time.time()
    train_loss = 0
    num_correct = 0
    num_correct2 = 0
    num_correct_joint = 0
    loop = tqdm(enumerate(train_loader), total=len(train_loader), desc='Train')
    running_smaple = 0
    loss_fct = get_loss_func(args.loss_type, sigma=args.sigma, weight=args.weight, num_classes=args.num_classes) if args.method != 'AggdNet_joint' else None
    for step, batch in loop:
        batch = tuple(t.to(args.device) for t in batch)
        x, y, y2 = batch
        if args.method == 'AggdNet_joint':
            logits, logits2, sigma_mat, sigma_mat2 = model(x)
            loss_fct = get_loss_func(args.loss_type, sigma_mat=sigma_mat, weight=args.weight, num_classes=args.num_classes)
            loss_fct2 = get_loss_func(args.loss_type, sigma_mat=sigma_mat2, weight=args.weight, num_classes=args.num_classes2)
            loss = loss_fct(logits, y) + loss_fct2(logits2, y2)
        elif args.method == 'ORNet_joint':
            logits, logits2, logits_r, logits_r2 = model(x)
            loss = loss_fct(logits, logits_r, y) + loss_fct(logits2, logits_r2, y2)
        else:
            logits, logits2 = model(x)
            loss = loss_fct(logits, y) + loss_fct(logits2, y2)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_loss += len(y) * loss.item()
        pred = logits.argmax(dim=1)
        pred2 = logits2.argmax(dim=1)
        num_correct += torch.eq(pred, y).sum().float().item()
        num_correct2 += torch.eq(pred2, y2).sum().float().item()
        num_correct_joint += num_consistent(pred, pred2, y.detach(), y2.detach())
        running_smaple += len(y)
        loop.set_description(f'(Train) Epoch [{current_epoch:<3d}/{args.epochs}]')
        info = {'loss': '{:.6f}'.format(train_loss/running_smaple), "joint_acc": '{:.4f}'.format(num_correct_joint/running_smaple)}
        loop.set_postfix(info)

    scheduler.step()
    results['joint_ac'] = num_correct_joint / len(train_loader.dataset)
    results['ac'] = num_correct / len(train_loader.dataset)
    results['ac2'] = num_correct2 / len(train_loader.dataset)
    results['loss'] = train_loss / len(train_loader.dataset)
    results['time'] = (time.time() - start) / 60
    return results


def train(args, model):
    best_tra_ac = 0
    best_val_ac = 0
    best_val_res = 0
    best_val_qwk = 0
    best_epoch = 0
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    # get dataset
    train_loader, val_loader, test_loader = get_loader(args)
    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.95,
                                weight_decay=args.weight_decay,
                                nesterov=args.nesterov)
    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=args.epochs, init_lr=args.learning_rate)
    elif args.decay_type == "step":
        scheduler = lr_scheduler.StepLR(optimizer, step_size=args.epochs//3, gamma=args.gamma)

    for epoch in range(args.epochs):
        train_res = train_one_epoch(args, model, train_loader, optimizer, scheduler, epoch)
        val_res = eval_one_epoch(args, model, val_loader, optimizer, scheduler, epoch)
        if epoch % args.eval_freq == 0 or epoch == args.epochs-1:
            print('Epoch [{:<3d}/{}], lr:{:.6f}, tra_loss:{:.4f}, joint_ac:{:.4f}'.format(train_res['ep'], args.epochs, train_res['lr'], train_res['loss'], train_res['joint_ac']), end='|')
            print('val_loss:{:.4f}, joint_ac:{:.4f}, runtime:{:.2f}m'.format(val_res['loss'], val_res['joint_ac'], train_res['time']+val_res['time']))
        if val_res['joint_ac'] >= best_val_ac:
            best_val_res = val_res
            best_val_ac = val_res['joint_ac']
            best_tra_ac = train_res['joint_ac']
            best_epoch = train_res['ep']
            # state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'epoch': best_epoch}
            path = os.path.join(args.output_dir, f'{args.name}_best.pth')
            torch.save(model.state_dict(), path)
    print(f'best_val_ac: {best_val_ac:.4f} on best_epoch: {best_epoch}, train_ac: {best_tra_ac:.4f}')
    print('-'*20 + 'Training end' + '-'*20)

    if test_loader is not None:
        print('Starting testing ......')
        model.load_state_dict(torch.load(path))
        if args.dataset == 'messidor':
            test_res = eval_messidor(args, model, test_loader, optimizer, scheduler)
            print('auc:{:.4f}, ac:{:.4f}, Pre:{:.4f}, Re:{:.4f}, F1:{:.4f}'.format(test_res['auc'], 
                                                test_res['ac'], test_res['pre'], test_res['rec'], test_res['f1']))
            
        else:
            test_res = eval_one_epoch(args, model, test_loader, optimizer, scheduler)
            test_joint_ac = test_res['joint_ac']
            test_ac = test_res['ac']
            test_ac2 = test_res['ac2']
            print(f'test_joint_ac:{test_joint_ac:.4f},test_ac1:{test_ac:.4f}, test_ac2:{test_ac2:.4f}')
    print('-'*60)
    print()


def main():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", required=False, default='none',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", default="idrid", help="Which downstream task.")
    parser.add_argument('--data_root', type=str, default='./datasets/idrid')
    parser.add_argument("--backbone", choices=['resnet18','resnet34','resnet50','vit_base_patch16_224','efficientnet_b1'], default="resnet18", help="Which variant to use.")
    parser.add_argument("--method", type=str, default="Base", help="Which method to use.")
    parser.add_argument("--task", type=str, default="joint", help="Which task to use.")
    parser.add_argument("--loss_type", type=str, default="cross_entropy_loss", help="Which loss function to use.")
    parser.add_argument("--pretrained_dir", type=str, default=None, help="Where to search for pretrained models.")
    parser.add_argument("--output_dir", default="output", type=str,
                        help="The train_resput directory where checkpoints will be written.")
    parser.add_argument("--img_size", default=896, type=int, help="Resolution size")
    parser.add_argument("--weight", default=0.1, type=float, help="weight")
    parser.add_argument("--sigma", default=1, type=float, help="varience of gauss distri.")
    parser.add_argument("--head_type", type=str, default='one_layer', help="head type")
    parser.add_argument("--var_type", type=str, default='one_layer', help="vp type")
    parser.add_argument("--train_batch_size", default=8, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=8, type=int,
                        help="Total batch size for eval.")
    parser.add_argument("--eval_freq", default=1, type=int,
                        help="Run prediction on validation set every epoch")
    parser.add_argument("--learning_rate", default=1e-3, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=1e-4, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--nesterov", action="store_true", help='whether use nesterov')
    parser.add_argument("--gamma", type=float, default=0.5)
    parser.add_argument("--n_fold", type=int, default=0, help='cross-validation setting.')
    parser.add_argument("--epochs", default=120, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--decay_type", choices=["cosine", "step"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument('--seed', type=int, default=100,
                        help="random seed for initialization")
    args = parser.parse_args()

    # Setup CUDA
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.device = device
    
    # Set seed
    set_seed(args)
    if args.task == 'joint':
        assert args.method.split('_')[-1] ==  'joint', 'method and task is not consitent.'
    else:
        assert args.method.split('_')[-1] != 'joint', 'method and task is not consitent.'
    # Model & Tokenizer Setup
    args, model = setup(args)

    if args.method == 'ORNet':
        args.loss_type = 'ornet_loss'
    elif args.method == 'AggdNet':
        args.loss_type = 'aggd_loss'
    print(f'using loss type: {args.loss_type}')
    print(args)
    print('starting train.....\n')

    # Training
    train(args, model)


if __name__ == "__main__":
    main()
