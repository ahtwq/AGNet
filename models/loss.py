import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import scipy
from functools import partial
from scipy.special import gamma


def get_loss_func(tag='cross_entropy', **kwargs):
    # default: classification with categorical cross-enropy
    if tag == 'cross_entropy_loss':
        return F.cross_entropy

    elif tag == 'norm_loss':
        return partial(norm_loss, sigma=kwargs['sigma'], weight=kwargs['weight'], num_classes=kwargs['num_classes'])

    elif tag == 'aggd_loss':
        return partial(aggd_loss, sigma_mat=kwargs['sigma_mat'], weight=kwargs['weight'], num_classes=kwargs['num_classes'])

    elif tag == 'ornet_loss':
        return partial(ornet_loss, num_classes=kwargs['num_classes'])
    
    elif tag == 'adaptive_ordinal_loss':
        return partial(adaptive_ordinal_loss, num_classes=kwargs['num_classes'])
    
    elif tag == 'soft_labels_loss':
        return soft_labels_loss
    
    elif tag == 'mean_variance_loss':
        return mean_variance_loss

    elif tag == 'unimodal_concentrated_loss':
        return unimodal_concentrated_loss_not_official

    else:
        raise NotImplementedError()


def norm_loss(logits: torch.tensor, labels: torch.tensor, sigma=1, weight=0.1, num_classes=5):
    """
    Args:
        logits (torch.tensor): Size(batch_size, nr_classes])
        labels (torch.tensor): Size([batch_size])
        sigma
    """
    def gen_norm_distribution(sigma=1, num_classes=5):
        SoftLabel = np.zeros((num_classes, num_classes), dtype=np.float32)
        class_label = np.arange(0, num_classes)
        for i in range(num_classes):
            slabel = scipy.stats.norm.pdf(class_label, i, sigma)
            SoftLabel[i] = slabel / slabel.sum()
        return SoftLabel

    SoftLabel = gen_norm_distribution(sigma=sigma, num_classes=num_classes)
    SoftLabel = torch.from_numpy(SoftLabel).to(labels.device)
    label_distributions = SoftLabel[labels[:, None]]
    
    prob_log = logits.log_softmax(dim=-1)
    dloss = torch.sum(-1 * prob_log * label_distributions, dim=-1).mean()

    loss = (1 - weight) * F.cross_entropy(logits, labels) + weight * dloss
    return loss


def ornet_loss(logits1: torch.tensor, logits2: torch.tensor, labels: torch.tensor, weight=0.125, num_classes=5):
    def get_reg_labels(num_classes=5):
        rlabels = torch.FloatTensor(np.arange(num_classes)+0.5)
        return rlabels

    Rlabels = get_reg_labels(num_classes)
    rlabels = Rlabels[labels[:, None]].to(logits1.device)
    loss = (1 - weight) * F.cross_entropy(logits1, labels, label_smoothing=0.1) + weight * F.smooth_l1_loss(logits2, rlabels)
    return loss


def aggd_loss(logits: torch.tensor, labels: torch.tensor, sigma_mat=1, weight=0.1, num_classes=5):
    """
    Args:
        logits (torch.tensor): Size(batch_size, nr_classes])
        labels (torch.tensor): Size([batch_size])
        sigma
    """
    def aggd(x, mu=0, left_variance=1, right_variance=1, alpha=2):
        """
        AGGD, Asymmetric Generalized Gaussian Distribution
        """
        gamma_1_alpha = gamma(1. / alpha)
        gamma_3_alpha = gamma(3. / alpha)
        left_beta = np.sqrt(left_variance * gamma_1_alpha / gamma_3_alpha)
        right_beta = np.sqrt(right_variance * gamma_1_alpha / gamma_3_alpha)
        factor = alpha / ((left_beta + right_beta) * gamma_1_alpha)
        return np.where(
            x - mu < 0, 
            factor * np.exp(-np.abs((x-mu) / left_beta)**alpha),
            factor * np.exp(-np.abs((x-mu) / right_beta)**alpha)
        )

    def gen_sysnorm_distribution(labels, sigma_mat, num_classes=5):
        SoftLabel = np.zeros((len(labels), num_classes), dtype=np.float32)
        class_label = np.arange(0, num_classes)
        for i in range(len(labels)):
            sigmas = sigma_mat[i]
            slabel = aggd(class_label, labels[i].item(), sigmas[0].item(), sigmas[1].item())
            SoftLabel[i] = slabel / slabel.sum()
        return SoftLabel
    
    SoftLabel = gen_sysnorm_distribution(labels, sigma_mat,num_classes)
    label_distributions = torch.from_numpy(SoftLabel).to(labels.device)
    prob_log = logits.log_softmax(dim=-1)
    dloss = torch.sum(-1 * prob_log * label_distributions, dim=-1).mean()
    
    loss = (1 - weight) * F.cross_entropy(logits, labels) + weight * dloss
    return loss



def adaptive_ordinal_loss(logits, labels, num_classes=5):
    def set_weights(num_classes):
        # # weight matrix 01 (wm01)
        # init_weights = np.array([[1, 2, 3, 4, 5],
        #                          [2, 1, 2, 3, 4],
        #                          [3, 2, 1, 2, 3],
        #                          [4, 3, 2, 1, 2],
        #                          [5, 4, 3, 2, 1]], dtype=np.float)

        # weight matrix 02 (wm02)
        if num_classes == 5:
            init_weights = np.array([[1, 3, 5, 7, 9],
                                     [3, 1, 3, 5, 7],
                                     [5, 3, 1, 3, 5],
                                     [7, 5, 3, 1, 3],
                                     [9, 7, 5, 3, 1]], dtype=np.float32)
        elif num_classes == 4:
            init_weights = np.array([[1, 3, 5, 7],
                                     [3, 1, 3, 5],
                                     [5, 3, 1, 3],
                                     [7, 5, 3, 1]], dtype=np.float32)
        # # weight matrix 03 (wm03)
        # init_weights = np.array([[1, 4, 7, 10, 13],
        #                          [4, 1, 4, 7, 10],
        #                          [7, 4, 1, 4, 7],
        #                          [10, 7, 4, 1, 4],
        #                          [13, 10, 7, 4, 1]], dtype=np.float)

        # # weight matrix 04 (wm04)
        # init_weights = np.array([[1, 3, 6, 7, 9],
        #                          [4, 1, 4, 5, 7],
        #                          [6, 4, 1, 3, 5],
        #                          [7, 5, 3, 1, 3],
        #                          [9, 7, 5, 3, 1]], dtype=np.float)

        adjusted_weights = init_weights + 1.0
        np.fill_diagonal(adjusted_weights, 0)

        return adjusted_weights

    cls_weights = set_weights(num_classes)
    prob_pred = torch.softmax(logits, dim=1)
    batch_num, class_num = logits.size()
    class_hot = np.zeros([batch_num, class_num], dtype=np.float32)
    labels_np = labels.data.cpu().numpy()
    for ind in range(batch_num):
        class_hot[ind, :] = cls_weights[labels_np[ind], :]
    class_hot = torch.from_numpy(class_hot).to(logits.device)
    class_hot = torch.autograd.Variable(class_hot)

    loss = torch.sum((prob_pred * class_hot)**2) / batch_num
    # loss = torch.mean(prob_pred * class_hot)

    return loss
    

def mean_variance_loss(logits: torch.tensor, labels: torch.tensor, lambda_1=0.2, lambda_2=0.05):
    """
    Computes the Mean-Variance Loss as defined in Mean-Variance Loss for Deep Age Estimation from a Face.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes])
        labels (torch.tensor): Size([batch_size])
        lambda_1 (float, optional): Weight of the mean loss. Defaults to 0.2.
        lambda_2 (float, optional): Weight of the variance loss. Defaults to 0.05.

    """
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1]

    probas = F.softmax(logits, dim=1)

    class_labels = torch.arange(0, nr_classes)
    class_labels = torch.broadcast_to(
        class_labels, probas.shape).to(labels.device)

    means = torch.sum(probas*class_labels, dim=1).to(labels.device)
    broadcast_means = torch.broadcast_to(
        means[:, None], probas.shape).to(labels.device)
    variances = torch.sum((broadcast_means - class_labels)
                          ** 2*probas, dim=1).to(labels.device)

    ce_loss = F.cross_entropy(logits, labels)
    mean_loss = torch.mean((means-labels)**2)/2.
    variance_loss = torch.mean(variances)

    loss = ce_loss + lambda_1*mean_loss + lambda_2*variance_loss
    return loss


def unimodal_concentrated_loss(logits: torch.tensor, labels: torch.tensor):
    """
    For the paper, we obtained the official implementation from the authors of Unimodal-Concentrated Loss. However, we were asked not to make it public. Therefore, the loss function has been removed from the release version of the code.
    """
    raise NotImplementedError(
        "For the paper, we obtained the official implementation from the authors of Unimodal-Concentrated Loss. However, we were asked not to make it public. Therefore, the loss function has been removed from the release version of the code.")


def unimodal_concentrated_loss_not_official(logits: torch.tensor, labels: torch.tensor, lambda_=10):
    """
    Computes the Unimodal-Concentrated Loss as defined in Unimodal-Concentrated Loss: Fully Adaptive Label Distribution Learning for Ordinal Regression.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes])
        labels (torch.tensor): Size([batch_size])
        lambda_ (float, optional): Weight of the unimodal loss. Defaults to 1000.

    """
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1]
    eps = 1e-10

    probas = F.softmax(logits, dim=1)
    diffs = torch.diff(probas, dim=1).to(labels.device)

    class_labels = torch.arange(0, nr_classes)
    class_labels = torch.broadcast_to(
        class_labels, probas.shape).to(labels.device)

    broadcast_labels = torch.broadcast_to(
        labels[:, None], probas.shape).to(labels.device)

    select = []
    for label in labels:
        select.append(torch.tensor(
            [i for i in range(label)] + [i for i in range(label+1, nr_classes)]))
    select = torch.stack(select).to(labels.device)

    sign = torch.gather(torch.sign(
        class_labels-broadcast_labels), 1, select).to(labels.device)

    signed_and_clipped_diffs = F.relu(sign*diffs).to(labels.device)
    unimodal_loss = torch.mean(
        torch.sum(signed_and_clipped_diffs, dim=1)).to(labels.device)

    means = torch.sum(probas*class_labels, dim=1).to(labels.device)
    broadcast_means = torch.broadcast_to(
        means[:, None], probas.shape).to(labels.device)
    variances = torch.sum(((broadcast_means - class_labels)**2)
                          * probas, dim=1).to(labels.device)

    assert torch.all(variances >= 0)

    concentrated_loss = torch.mean(
        torch.log(2*np.pi*variances + eps) / 2. + ((means-labels)**2)/(2.*variances + eps))
    
    loss = concentrated_loss + lambda_ * unimodal_loss
    return loss


def soft_labels_loss(logits: torch.tensor, labels: torch.tensor, distance_squared=False):
    """
    Computes the loss as defined in Soft Labels for Ordinal Regression.

    Args:
        logits (torch.tensor): Size(batch_size, nr_classes])
        labels (torch.tensor): Size([batch_size])
        distance_squared (bool, optional): If True, measures label distance as L2 instead of L1 for generating label distribution. Defaults to False.

    """
    batch_size = logits.shape[0]
    nr_classes = logits.shape[1]

    def abs_diff(a, b):
        return torch.abs(a-b)

    def squared_diff(a, b):
        return (a-b)**2

    def distance_measure(a, b):
        if distance_squared:
            return squared_diff(a, b)
        else:
            return abs_diff(a, b)

    probas = F.softmax(logits, dim=1)
    diffs = torch.diff(probas, dim=1)

    class_labels = torch.arange(0, nr_classes)
    class_labels = torch.broadcast_to(
        class_labels, probas.shape).to(labels.device)
    broadcast_labels = torch.broadcast_to(
        labels[:, None], probas.shape).to(labels.device)

    label_distributions = - \
        distance_measure(class_labels, broadcast_labels).type(logits.dtype)
    label_distributions = torch.softmax(label_distributions, dim=1)

    prob_log = logits.log_softmax(dim=-1)
    loss = torch.sum(-1 * prob_log * label_distributions, dim=-1).mean()
    return loss


def mae_loss(logits: torch.tensor, labels: torch.tensor):
    """
    Computes MAE loss for a regression network.

    Args:
        logits (torch.tensor): Size(batch_size, 1]) Regression outputs.
        labels (torch.tensor): Size([batch_size]) 

    """
    loss = torch.mean(torch.abs(logits - labels[:, None].float()))
    return loss
