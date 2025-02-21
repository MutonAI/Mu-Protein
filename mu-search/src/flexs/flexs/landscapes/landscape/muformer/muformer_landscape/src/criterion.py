import torch
import torch.nn.functional as F

def pearson_loss(x, y):
	mean_x = torch.mean(x)
	mean_y = torch.mean(y)
	xm = x.sub(mean_x)
	ym = y.sub(mean_y)
	r_num = xm.dot(ym)
	r_den = torch.norm(xm, 2) * torch.norm(ym, 2)
	r_val = r_num / r_den
	return 1 - r_val

def pearson_correlation_loss(y_true, y_pred, normalized=False):
    """
     Calculate pearson correlation loss   
    :param y_true: distance matrix tensor tensor size (batch_size, batch_size)
    :param y_pred: distance matrix tensor tensor size (batch_size, batch_size)
    :param normalized: if True, Softmax is applied to the distance matrix
    :return: loss tensor
    """
    if normalized:
        y_true = F.softmax(y_true, axis=-1)
        y_pred = F.softmax(y_pred, axis=-1)

    sum_true = torch.sum(y_true)
    sum2_true = torch.sum(torch.pow(y_true, 2))                         # square ~= np.pow(a,2)

    sum_pred = torch.sum(y_pred)
    sum2_pred = torch.sum(torch.pow(y_pred, 2))

    prod = torch.sum(y_true * y_pred)
    n = y_true.shape[0]                                                     # n == y_true.shape[0]

    corr = n * prod - sum_true * sum_pred
    corr /= torch.sqrt(n * sum2_true - sum_true * sum_true + torch.finfo(torch.float32).eps)       # K.epsilon() == 1e-7
    corr /= torch.sqrt(n * sum2_pred - sum_pred * sum_pred + torch.finfo(torch.float32).eps)

    return 1 - corr

