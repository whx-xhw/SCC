import torch
import torch.nn.functional as F
import torchvision.transforms
import numpy as np
import tools_usps
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
import sep_dataset_usps


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


class SemiL(object):
    def __call__(self, outputs_x, targets_x):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        return Lx


def warm_up(model, optim, epoch, loss1, loss2, dataloader, device, crit):
    model.train()
    for _ in range(epoch):
        for idx, (x, y) in enumerate(dataloader):
            x = x.to(device)
            y = y.to(device)
            x = x.view(-1, 256)
            x = x.float()
            optim.zero_grad()
            out = model(x)
            L1 = loss1(out, y.long())
            L2 = loss2(out)
            L = L1 + L2 * crit
            L.backward()
            optim.step()
    return model


def eval(model, data_loader, device, loss, eval_number):
    model.eval()
    losses = np.zeros(shape=(eval_number,))
    with torch.no_grad():
        for idx, (x, y) in enumerate(data_loader):
            x = x.to(device)
            y = y.to(device)
            x = x.view(-1, 256)
            x = x.float()
            x_np = x.detach().cpu().numpy()
            out = model(x)
            loss_ = loss(out, y.long())
            loss_np = loss_.detach().cpu().numpy()
            if x_np.shape[0] < 10000:
                losses[idx * 10000: idx * 10000 + x_np.shape[0]] = loss_np
            else:
                losses[idx * 10000: (idx + 1) * 10000] = loss_np
    losses = (losses - np.min(losses)) / (np.max(losses) - np.min(losses))
    input_loss = losses.reshape(-1, 1)
    gmm = GaussianMixture(n_components=2, max_iter=1000, tol=1e-3, reg_covar=5e-4)
    gmm.fit(input_loss)
    prob = gmm.predict_proba(input_loss)
    prob = prob[:, gmm.means_.argmin()]
    return prob, losses


def ensemble_selector(prob, aug_times, upsilon, pseudo_label, ground_truth_fixed, classifier_num, epoch):
    labeled_index = []
    unlabeled_index = []
    w_b = []
    for i in range(int(prob.shape[0] / aug_times)):
        p_stage = prob[i * aug_times: (i + 1) * aug_times]
        p_deter = p_stage >= upsilon
        p_deter.astype(int)
        mean_decision = np.mean(p_deter)
        if mean_decision >= 0.5:
            labeled_index.append(i)
            w_b.append(np.mean(p_stage))
        else:
            unlabeled_index.append(i)

    labeled_pseudo_label = pseudo_label[labeled_index]
    labeled_ground_truth_fixed = ground_truth_fixed[labeled_index]
    count = 0
    for j in range(labeled_pseudo_label.shape[0]):
        if labeled_pseudo_label[j] == labeled_ground_truth_fixed[j]:
            count += 1
    labeled_select_rate = len(labeled_index) / (prob.shape[0] / aug_times)
    labeled_acc = count / len(labeled_index)

    Wrong_idx = []
    for i in range(pseudo_label.shape[0]):
        if pseudo_label[i] != ground_truth_fixed[i]:
            Wrong_idx.append(i)

    select_idx_wrong = list(set(labeled_index) & set(Wrong_idx))
    wrong_find_rate = 1 - len(select_idx_wrong) / len(Wrong_idx)
    print('Epoch: {}, dual model{}, select rate:{:.4f}, select acc:{:.4f}, misassignments detecting rate:{:.4f}'.format(
        epoch, classifier_num, labeled_select_rate, labeled_acc, wrong_find_rate))
    return labeled_index, unlabeled_index, w_b


def labeled_and_unlabeled_loader(absolute_labeled_information, absolute_unlabeled_information):
    absolute_labeled_index = absolute_labeled_information[0]
    labeled_pseudo_label = absolute_labeled_information[1]

    absolute_unlabeled_index = absolute_unlabeled_information[0]
    unlabeled_pseudo_label = absolute_unlabeled_information[1]

    trans = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    labeled_set = sep_dataset_usps.USPS(root='./data/', transform=trans, select_idx=absolute_labeled_index,
                                        pseudo_labels=labeled_pseudo_label)

    unlabeled_set = sep_dataset_usps.USPS(root='./data/', transform=trans, select_idx=absolute_unlabeled_index,
                                          pseudo_labels=unlabeled_pseudo_label)

    labeled_loader = DataLoader(dataset=labeled_set, batch_size=5000, shuffle=False, drop_last=False)
    unlabeled_loader = DataLoader(dataset=unlabeled_set, batch_size=5000, shuffle=False, drop_last=False)

    return labeled_loader, unlabeled_loader


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def test(model1, model2, raw_data, ground_truth, device, epoch):
    raw_data = raw_data.to(device)
    raw_data = raw_data.view(-1, 256)
    raw_data = raw_data.float()

    out1 = model1(raw_data)
    out2 = model2(raw_data)

    pre_membership = torch.softmax(out1, dim=1) + torch.softmax(out2, dim=1)
    pre_membership = pre_membership.detach().cpu().numpy()
    pre_label = np.argmax(pre_membership, axis=1)
    acc, per_class_num, nmi, ari = tools_usps.clustering_metrix(true_label=ground_truth, pre_label=pre_label)
    print('Epoch: {}, warm up acc:{}'.format(epoch, acc))


def input_target_align(labeled_aug_data, unlabeled_aug_data, target_labeled, target_unlabeled, aug_times):
    aug_data = np.concatenate([labeled_aug_data, unlabeled_aug_data], axis=0)
    aug_data = np.reshape(aug_data, newshape=(-1, 16, 16))
    target_labeled_aligned = np.zeros(shape=(target_labeled.shape[0] * aug_times, 10))
    target_unlabeled_aligned = np.zeros(shape=(target_unlabeled.shape[0] * aug_times, 10))
    for i in range(target_labeled.shape[0]):
        target_labeled_aligned[i * aug_times: (i + 1) * aug_times] = target_labeled[i]
    for i in range(target_unlabeled.shape[0]):
        target_unlabeled_aligned[i * aug_times: (i + 1) * aug_times] = target_unlabeled[i]
    target = np.concatenate([target_labeled_aligned, target_unlabeled_aligned], axis=0)
    return aug_data, target


def idx_align(idx, aug_times):
    aligned = np.zeros(shape=(idx.shape[0] * aug_times), dtype=np.int64)
    for i in range(idx.shape[0]):
        ii = np.arange(aug_times) + idx[i] * aug_times
        aligned[i * aug_times: (i + 1) * aug_times] = ii
    return aligned


def aug_align(aug_data, idx, aug_times):
    aligned_idx = idx_align(idx, aug_times)
    aug_data_select = aug_data[aligned_idx]
    aug_data_select = np.reshape(aug_data_select, newshape=(-1, 16, 16))
    return aug_data_select


def sample_beta_distribution():
    sampling_times = 300
    beta_distribution_alpha = 0.3
    beta_distribution_beta = 0.3
    mixed_weight_ = np.zeros(shape=(sampling_times,))
    for i in range(sampling_times):
        sampled_weight = np.random.beta(beta_distribution_alpha, beta_distribution_beta)
        sampled_weight_ = 1 - sampled_weight
        mixed_weight_[i] = np.max(np.array([sampled_weight, sampled_weight_]))

    sampled = mixed_weight_
    return np.mean(sampled)


def mixmatch(model1, model2, optim, labeled_loader, unlabeled_loader, batch_size, device, temp, semi_loss, w_b,
             aug_data, tr_sel_idx, tr_unsel_idx, aug_times, sigma):
    model1.train()
    model2.eval()

    wb = np.array(w_b)

    pb = np.zeros(shape=(tr_sel_idx.shape[0], 10))
    labeled_pseudo_y_np = np.zeros(shape=(tr_sel_idx.shape[0],), dtype=np.int64)

    for idx, (labeled_x, labeled_pseudo_y, labeled_idx) in enumerate(labeled_loader):
        labeled_idx_np = labeled_idx.detach().cpu().numpy()
        labeled_pseudo_y_bs = labeled_pseudo_y.detach().cpu().numpy()
        tr_labeled_idx_np = tr_sel_idx[labeled_idx_np]
        labeled_aug_data = aug_align(aug_data=aug_data, idx=tr_labeled_idx_np, aug_times=aug_times)
        labeled_aug_data_tensor = torch.tensor(labeled_aug_data).to(device)
        labeled_aug_data_tensor = labeled_aug_data_tensor.view(-1, 256)
        labeled_aug_data_tensor = labeled_aug_data_tensor.float()
        out = model1(labeled_aug_data_tensor)
        out_np = out.detach().cpu().numpy()
        pb_all = out_np
        pb_bs = np.zeros(shape=(labeled_aug_data.shape[0] // aug_times, 10))
        for ii in range(labeled_aug_data.shape[0] // aug_times):
            soft = softmax(pb_all[ii * aug_times: (ii + 1) * aug_times].T)
            pb_bs[ii] = np.sum(soft.T, axis=0) / aug_times
        if labeled_idx_np.shape[0] < batch_size:
            pb[idx * batch_size: idx * batch_size + labeled_idx_np.shape[0]] = pb_bs
            labeled_pseudo_y_np[idx * batch_size: idx * batch_size + labeled_idx_np.shape[0]] = labeled_pseudo_y_bs
        else:
            pb[idx * batch_size: (idx + 1) * batch_size] = pb_bs
            labeled_pseudo_y_np[idx * batch_size: (idx + 1) * batch_size] = labeled_pseudo_y_bs
    pb = torch.tensor(pb)
    wb = np.reshape(wb, newshape=(-1, 1))
    wb = torch.tensor(wb)
    labeled_pseudo_y_tensor = torch.tensor(labeled_pseudo_y_np)
    labeled_pseudo_y_tensor = torch.zeros(int(tr_sel_idx.shape[0]), 10).scatter_(1, labeled_pseudo_y_tensor.view(-1, 1), 1)
    pb = wb * labeled_pseudo_y_tensor + (1 - wb) * pb
    pb_sharpen = pb ** (1 / temp)
    # pb_sharpen = torch.tensor(pb_sharpen)
    target_labeled = pb_sharpen / pb_sharpen.sum(dim=1, keepdim=True)
    target_labeled = target_labeled.detach()

    qu = np.zeros(shape=(tr_unsel_idx.shape[0], 10))
    for idx, (unlabeled_x, unlabeled_pseudo_y, unlabeled_idx) in enumerate(unlabeled_loader):
        unlabeled_idx_np = unlabeled_idx.detach().cpu().numpy()
        tr_unlabeled_idx_np = tr_unsel_idx[unlabeled_idx_np]
        unlabeled_aug_data = aug_align(aug_data=aug_data, idx=tr_unlabeled_idx_np, aug_times=aug_times)
        unlabeled_aug_data_tensor = torch.tensor(unlabeled_aug_data).to(device)
        unlabeled_aug_data_tensor = unlabeled_aug_data_tensor.view(-1, 256)
        unlabeled_aug_data_tensor = unlabeled_aug_data_tensor.float()
        out1 = model1(unlabeled_aug_data_tensor)
        out2 = model2(unlabeled_aug_data_tensor)
        qu_all_model1 = out1.detach().cpu().numpy()
        qu_all_model2 = out2.detach().cpu().numpy()
        qu_all_model1 = torch.tensor(qu_all_model1)
        qu_all_model2 = torch.tensor(qu_all_model2)

        qu_bs = np.zeros(shape=(unlabeled_aug_data.shape[0] // aug_times, 10))
        for ii in range(unlabeled_aug_data.shape[0] // 20):
            qu_soft1 = torch.sum(torch.softmax(qu_all_model1[ii * aug_times: (ii + 1) * aug_times], dim=1), dim=0)
            qu_soft2 = torch.sum(torch.softmax(qu_all_model2[ii * aug_times: (ii + 1) * aug_times], dim=1), dim=0)
            qu_bs_k = (qu_soft1 + qu_soft2) / (2 * aug_times)
            qu_bs[ii] = qu_bs_k.detach().cpu().numpy()

        if unlabeled_idx_np.shape[0] < batch_size:
            qu[idx * batch_size: idx * batch_size + unlabeled_idx_np.shape[0]] = qu_bs

        else:
            qu[idx * batch_size: (idx + 1) * batch_size] = qu_bs

    qu = torch.tensor(qu)
    qu_sharpen = qu ** (1 / temp)

    Lkkk = nn.MSELoss()

    target_unlabeled = qu_sharpen / qu_sharpen.sum(dim=1, keepdim=True)
    target_unlabeled = target_unlabeled.detach()

    mix_lamb = sample_beta_distribution()
    aligned_tr_sel_idx = idx_align(tr_sel_idx, aug_times=aug_times)
    aligned_tr_unsel_idx = idx_align(tr_unsel_idx, aug_times=aug_times)

    labeled_aug_data = aug_data[aligned_tr_sel_idx]
    unlabeled_aug_data = aug_data[aligned_tr_unsel_idx]

    aug_da, aug_tar = input_target_align(labeled_aug_data=labeled_aug_data, unlabeled_aug_data=unlabeled_aug_data,
                                         target_labeled=target_labeled, target_unlabeled=target_unlabeled,
                                         aug_times=aug_times)
    aug_da_tensor = torch.tensor(aug_da)
    aug_tar_tensor = torch.tensor(aug_tar)

    randim_idx = torch.randperm(aug_da_tensor.size(0))

    input_a, input_b = aug_da_tensor, aug_da_tensor[randim_idx]
    target_a, target_b = aug_tar_tensor, aug_tar_tensor[randim_idx]

    mixed_input = mix_lamb * input_a + (1 - mix_lamb) * input_b
    mixed_target = mix_lamb * target_a + (1 - mix_lamb) * target_b

    mixed_input0 = mixed_input[:int(tr_sel_idx.shape[0]) * aug_times]
    mixed_target0 = mixed_target[:int(tr_sel_idx.shape[0]) * aug_times]

    mixed_input1 = mixed_input[int(tr_sel_idx.shape[0]) * aug_times:]
    mixed_target1 = mixed_target[int(tr_sel_idx.shape[0]) * aug_times:]

    sub_train_set0 = TensorDataset(mixed_input0, mixed_target0)
    sub_train_set1 = TensorDataset(mixed_input1, mixed_target1)
    sub_train_loader0 = DataLoader(dataset=sub_train_set0, batch_size=256, shuffle=True, drop_last=False)
    sub_train_loader1 = DataLoader(dataset=sub_train_set1, batch_size=256, shuffle=True, drop_last=False)

    L_ = 0.
    unlabeled_train_iter = iter(sub_train_loader1)
    for idd, (x1, y1) in enumerate(sub_train_loader0):
        try:
            x2, y2 = next(unlabeled_train_iter)
        except:
            unlabeled_train_iter = iter(sub_train_loader1)
            x2, y2 = next(unlabeled_train_iter)
        x1 = x1.to(device)
        y1 = y1.to(device)
        x2 = x2.to(device)
        y2 = y2.to(device)
        x1 = x1.view(-1, 256)
        x1 = x1.float()
        x2 = x2.view(-1, 256)
        x2 = x2.float()
        y1 = y1.float()
        y2 = y2.float()
        logits = model1(x1)
        L_x = semi_loss(logits, y1)

        logits2 = model1(x2)
        L_u = Lkkk(logits2, y2)
        L0 = L_x + L_u * sigma

        prior = torch.ones(10) / aug_times
        prior = prior.to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = L0 + penalty

        optim.zero_grad()
        loss.backward()
        optim.step()

        L_ += loss.item()

    return model1

