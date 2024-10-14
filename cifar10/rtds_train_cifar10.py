import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np
import tools_cifar10
from sklearn.mixture import GaussianMixture
from torch.utils.data import DataLoader, TensorDataset
from torchvision import transforms
import cv2
import torch.nn as nn
import sep_dataset_cifar10


class NegEntropy(object):
    def __call__(self, outputs):
        probs = torch.softmax(outputs, dim=1)
        return torch.mean(torch.sum(probs.log() * probs, dim=1))


class SemiL(object):
    def __call__(self, outputs_x, targets_x):
        Lx = -torch.mean(torch.sum(F.log_softmax(outputs_x, dim=1) * targets_x, dim=1))
        return Lx


def warm_up(model, optim, epoch, loss1, loss2, dataloader, device, crit, aug_times, bs):

    for epo in range(epoch):
        if epo == epoch - 1:
            model.train()
            datas = np.zeros(shape=(len(dataloader.dataset) * aug_times, 3, 32, 32))
            pseudo_labelss = np.zeros(shape=(len(dataloader.dataset) * aug_times,), dtype=np.int32)
            fixed_labelss = np.zeros(shape=(len(dataloader.dataset) * aug_times,), dtype=np.int32)
            index_ = np.zeros(shape=(len(dataloader.dataset) * aug_times,), dtype=np.int32)
            for idx, (x, y, f_t, s_idx) in enumerate(tqdm(dataloader)):
                x = torch.cat(x, dim=0)
                y = torch.cat(y, dim=0)
                f_t = torch.cat(f_t, dim=0)
                s_idx = torch.cat(s_idx, dim=0)

                x = x.to(device)
                y = y.to(device)

                out = model(x)
                L1 = loss1(out, y.long())
                L2 = loss2(out)
                L = L1 + L2 * crit
                optim.zero_grad()
                L.backward()
                optim.step()

                x_np = x.detach().cpu().numpy()
                y_np = y.detach().cpu().numpy()
                f_t_np = f_t.detach().cpu().numpy()
                s_idx_np = s_idx.detach().cpu().numpy()

                datas[idx * bs * aug_times: idx * bs * aug_times + x_np.shape[0]] = x_np
                pseudo_labelss[idx * bs * aug_times: idx * bs * aug_times + x_np.shape[0]] = y_np
                fixed_labelss[idx * bs * aug_times: idx * bs * aug_times + x_np.shape[0]] = f_t_np
                index_[idx * bs * aug_times: idx * bs * aug_times + x_np.shape[0]] = s_idx_np

        else:
            model.train()
            for idx, (x, y, _, _) in enumerate(dataloader):
                x = torch.cat(x, dim=0)
                y = torch.cat(y, dim=0)
                x = x.to(device)
                y = y.to(device)

                optim.zero_grad()
                out = model(x)
                L1 = loss1(out, y.long())
                L2 = loss2(out)
                L = L1 + L2 * crit
                L.backward()
                optim.step()

    return model, datas, pseudo_labelss, fixed_labelss, index_


def warm_up_ntm(model, optim, epoch, loss1, loss2, dataloader, device, crit):
    for _ in range(epoch):
        model.train()
        for idx, (x, y, _, _) in enumerate(dataloader):
            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)
            x = x.to(device)
            y = y.to(device)

            optim.zero_grad()
            out = model(x)
            L1 = loss1(out, y.long())
            L2 = loss2(out)
            L = L1 + L2 * crit
            L.backward()
            optim.step()

    return model


def test(model1, model2, loader, device, epoch, bs):
    membership1 = np.zeros(shape=(len(loader.dataset), 10))
    membership2 = np.zeros(shape=(len(loader.dataset), 10))
    raw_label = np.zeros(shape=(len(loader.dataset), ), dtype=np.int32)

    for idx, (x, y) in enumerate(loader):
        x = x.to(device)
        y = y.to(device)

        y_np = y.detach().cpu().numpy()
        raw_label[bs * idx: bs * (idx + 1)] = y_np

        pred1 = model1(x)
        pred2 = model2(x)

        pred_np1 = pred1.detach().cpu().numpy()
        pred_np2 = pred2.detach().cpu().numpy()

        membership1[bs * idx: bs * (idx + 1)] = pred_np1
        membership2[bs * idx: bs * (idx + 1)] = pred_np2

    membership = membership1 + membership2
    pre_label = np.argmax(membership, axis=1)
    acc, per_class_num, nmi, ari = tools_cifar10.clustering_metrix(true_label=raw_label, pre_label=pre_label)
    print('Epoch: {}, warm up acc:{}'.format(epoch, acc))


def eval(model, device, aug_times, bs, data4eval, pseudo_labels4eval, fixed_labels4eval, s_idx4eval):
    data = data4eval
    pseudo_labels = pseudo_labels4eval
    fixed_labels = fixed_labels4eval
    s_idx = s_idx4eval

    data_tensor = torch.tensor(data).float()
    pseudo_labels_tensor = torch.tensor(pseudo_labels).long()
    fixed_labels_tensor = torch.tensor(fixed_labels).long()
    s_idx_tensor = torch.tensor(s_idx).long()

    num = int(data.shape[0] / aug_times)

    set = TensorDataset(data_tensor, pseudo_labels_tensor, fixed_labels_tensor, s_idx_tensor)
    loader = DataLoader(set, batch_size=bs, shuffle=False, num_workers=16, drop_last=False)

    pred = np.zeros(shape=(num * aug_times, 10))
    pseudo_labels = np.zeros(shape=(num * aug_times,), dtype=np.int32)
    loss_ = np.zeros(shape=(num * aug_times,))
    correct_labels = np.zeros(shape=(num * aug_times,), dtype=np.int32)
    idx_s = np.zeros(shape=(num * aug_times,), dtype=np.int32)
    crit_ = nn.CrossEntropyLoss(reduction='none')

    for idx, (x, p_t, f_t, idd) in enumerate(loader):
        x, p_t, f_t, idd = x.to(device), p_t.to(device), f_t.to(device), idd.to(device)
        output = model(x)
        loss = crit_(output, p_t)

        out_np = output.detach().cpu().numpy()
        p_t_np = p_t.detach().cpu().numpy()
        loss_np = loss.detach().cpu().numpy()
        f_t_np = f_t.detach().cpu().numpy()
        idd_np = idd.detach().cpu().numpy()

        pred[idx * bs: idx * bs + out_np.shape[0]] = out_np
        pseudo_labels[idx * bs: idx * bs + out_np.shape[0]] = p_t_np
        loss_[idx * bs: idx * bs + out_np.shape[0]] = loss_np
        correct_labels[idx * bs: idx * bs + out_np.shape[0]] = f_t_np
        idx_s[idx * bs: idx * bs + out_np.shape[0]] = idd_np

    loss_ = (loss_ - np.min(loss_)) / (np.max(loss_) - np.min(loss_))

    prob_list = []
    class_num_diff = []

    for _ in range(10): # avoid trivial solution of GMM
        input_loss = loss_.reshape(-1, 1)
        gmm = GaussianMixture(n_components=2, max_iter=1000, tol=1e-4, reg_covar=5e-4, init_params='kmeans')
        gmm.fit(input_loss)
        prob = gmm.predict_proba(input_loss)

        prob_hard = np.argmax(prob, axis=1)
        prob_hard_0 = np.where(prob_hard == 0)[0]
        prob_hard_1 = np.where(prob_hard == 1)[0]

        class_num_diff.append(np.abs(prob_hard_0.shape[0] - prob_hard_1.shape[0]))
        prob_list.append(prob)

    max_diff_idx = np.argmax(np.array(class_num_diff))

    prob = prob_list[max_diff_idx]
    prob_hard = np.argmax(prob, axis=1)
    prob_hard_0 = np.where(prob_hard == 0)[0]
    prob_hard_1 = np.where(prob_hard == 1)[0]
    select_cluster = np.argmax(np.array([prob_hard_0.shape[0], prob_hard_1.shape[0]]))
    prob = prob[:, select_cluster]

    return prob, pseudo_labels, correct_labels, idx_s


def ensemble_selector(prob, aug_times, bs, pseudo_labels, fixed_labels, idx_s, threshold, epoch, number, select_idx):
    num = select_idx.shape[0]
    prob_ = np.zeros(shape=(num, ))
    ps_t = np.zeros(shape=(num, ), dtype=np.int32)
    fi_t = np.zeros(shape=(num, ), dtype=np.int32)
    id_t = np.zeros(shape=(num, ), dtype=np.int32)

    for i in range(num // bs):
        prob_batch = prob[i * bs * aug_times: (i + 1) * bs * aug_times]
        pseudo_labels_batch = pseudo_labels[i * bs * aug_times: (i + 1) * bs * aug_times]
        fixed_labels_batch = fixed_labels[i * bs * aug_times: (i + 1) * bs * aug_times]
        idx_s_batch = idx_s[i * bs * aug_times: (i + 1) * bs * aug_times]

        prob_batch_list = []
        pseudo_labels_batch_list = []
        fixed_labels_batch_list = []
        idx_s_batch_list = []
        for j in range(aug_times):
            prob_batch_list.append(np.expand_dims(prob_batch[bs * j: bs * (j + 1)], axis=1))
            pseudo_labels_batch_list.append(np.expand_dims(pseudo_labels_batch[bs * j: bs * (j + 1)], axis=1))
            fixed_labels_batch_list.append(np.expand_dims(fixed_labels_batch[bs * j: bs * (j + 1)], axis=1))
            idx_s_batch_list.append(np.expand_dims(idx_s_batch[bs * j: bs * (j + 1)], axis=1))

        prob_batch_np = np.concatenate(prob_batch_list, axis=1)
        pseudo_labels_batch_np = np.concatenate(pseudo_labels_batch_list, axis=1)
        fixed_labels_batch_np = np.concatenate(fixed_labels_batch_list, axis=1)
        idx_s_batch_np = np.concatenate(idx_s_batch_list, axis=1)

        prob_batch_mean = np.mean(prob_batch_np, axis=1)
        pseudo_labels_batch_mean = np.mean(pseudo_labels_batch_np, axis=1)
        fixed_labels_batch_mean = np.mean(fixed_labels_batch_np, axis=1)
        idx_s_batch_mean = np.mean(idx_s_batch_np, axis=1)

        pseudo_labels_batch_mean = np.array(pseudo_labels_batch_mean, dtype=np.int32)
        fixed_labels_batch_mean = np.array(fixed_labels_batch_mean, dtype=np.int32)
        idx_s_batch_mean = np.array(idx_s_batch_mean, dtype=np.int32)

        prob_[i * bs: (i + 1) * bs] = prob_batch_mean
        ps_t[i * bs: (i + 1) * bs] = pseudo_labels_batch_mean
        fi_t[i * bs: (i + 1) * bs] = fixed_labels_batch_mean
        id_t[i * bs: (i + 1) * bs] = idx_s_batch_mean

    prob_batch_list = []
    pseudo_labels_batch_list = []
    fixed_labels_batch_list = []
    idx_s_batch_list = []

    prob_batch = prob[bs * (num // bs) * aug_times:]
    pseudo_labels_batch = pseudo_labels[bs * (num // bs) * aug_times:]
    fixed_labels_batch = fixed_labels[bs * (num // bs) * aug_times:]
    idx_s_batch = idx_s[bs * (num // bs) * aug_times:]

    remain = num - bs * (num // bs)

    for j in range(aug_times):
        prob_batch_list.append(np.expand_dims(prob_batch[remain * j: remain * (j + 1)], axis=1))
        pseudo_labels_batch_list.append(np.expand_dims(pseudo_labels_batch[remain * j: remain * (j + 1)], axis=1))
        fixed_labels_batch_list.append(np.expand_dims(fixed_labels_batch[remain * j: remain * (j + 1)], axis=1))
        idx_s_batch_list.append(np.expand_dims(idx_s_batch[remain * j: remain * (j + 1)], axis=1))

    prob_batch_np = np.concatenate(prob_batch_list, axis=1)
    pseudo_labels_batch_np = np.concatenate(pseudo_labels_batch_list, axis=1)
    fixed_labels_batch_np = np.concatenate(fixed_labels_batch_list, axis=1)
    idx_s_batch_np = np.concatenate(idx_s_batch_list, axis=1)

    prob_batch_mean = np.mean(prob_batch_np, axis=1)
    pseudo_labels_batch_mean = np.mean(pseudo_labels_batch_np, axis=1)
    fixed_labels_batch_mean = np.mean(fixed_labels_batch_np, axis=1)
    idx_s_batch_mean = np.mean(idx_s_batch_np, axis=1)

    pseudo_labels_batch_mean = np.array(pseudo_labels_batch_mean, dtype=np.int32)
    fixed_labels_batch_mean = np.array(fixed_labels_batch_mean, dtype=np.int32)
    idx_s_batch_mean = np.array(idx_s_batch_mean, dtype=np.int32)

    prob_[bs * (num // bs):] = prob_batch_mean
    ps_t[bs * (num // bs):] = pseudo_labels_batch_mean
    fi_t[bs * (num // bs):] = fixed_labels_batch_mean
    id_t[bs * (num // bs):] = idx_s_batch_mean

    labeled_index = id_t[np.where(prob_ > threshold)[0]]
    labeled_pseudo_label = ps_t[np.where(prob_ > threshold)[0]]
    labeled_ground_truth_fixed = fi_t[np.where(prob_ > threshold)[0]]
    labeled_p = prob_[np.where(prob_ > threshold)[0]]

    unlabeled_index = id_t[np.where(prob_ <= threshold)[0]]
    unlabeled_pseudo_label = ps_t[np.where(prob_ <= threshold)[0]]
    unlabeled_ground_truth_fixed = fi_t[np.where(prob_ <= threshold)[0]]
    unlabeled_p = prob_[np.where(prob_ <= threshold)[0]]

    count = 0
    for j in range(labeled_pseudo_label.shape[0]):
        if labeled_pseudo_label[j] == labeled_ground_truth_fixed[j]:
            count += 1
    labeled_select_rate = len(labeled_index) / (prob.shape[0] / aug_times)
    labeled_acc = count / len(labeled_index)

    Wrong_idx = []
    for i in range(ps_t.shape[0]):
        if ps_t[i] != fi_t[i]:
            Wrong_idx.append(i)

    select_idx_wrong = list(set(labeled_index) & set(Wrong_idx))
    wrong_find_rate = 1 - len(select_idx_wrong) / len(Wrong_idx)
    print('Epoch: {}, dual model{}, select rate:{:.4f}, select acc:{:.4f}, misassignments detecting rate:{:.4f}'.format(epoch, number, labeled_select_rate, labeled_acc, wrong_find_rate))

    tools_cifar10.select_samples_detailed_info(class_num=10, select_idx=np.where(prob_ > threshold)[0], pseudo_labels=labeled_pseudo_label, ground_truth_fixed=fi_t)

    return labeled_index, labeled_pseudo_label, labeled_p, unlabeled_index, unlabeled_pseudo_label, unlabeled_p


def labeled_and_unlabeled_loader(select_info, param):
    ground_truth_fixed = np.load('./data/ground_truth_fixed.npy')
    labeled_index = select_info[0]
    labeled_pseudo_label = select_info[1]
    unlabeled_index = select_info[3]
    unlabeled_pseudo_label = select_info[4]

    train_trans = get_train_trans()
    labeled_dataset = sep_dataset_cifar10.cifar10_for_warm_up(root='./data', transform=train_trans,
                                                              pseudo_labels=labeled_pseudo_label, fixed_labels=ground_truth_fixed,
                                                              select_index=labeled_index, aug_times=param.aug_times)

    unlabeled_dataset = sep_dataset_cifar10.cifar10_for_warm_up(root='./data', transform=train_trans,
                                                                pseudo_labels=unlabeled_pseudo_label, fixed_labels=ground_truth_fixed,
                                                                select_index=unlabeled_index, aug_times=param.aug_times)

    labeled_loader = DataLoader(labeled_dataset, batch_size=100, shuffle=False, num_workers=16, drop_last=False)
    unlabeled_loader = DataLoader(unlabeled_dataset, batch_size=100, shuffle=False, num_workers=16, drop_last=False)

    return labeled_loader, unlabeled_loader


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


def sample_beta_distribution():
    sampling_times = 300
    beta_distribution_alpha = 0.3
    beta_distribution_beta = 0.3
    mixed_weight_ = np.zeros(shape=(sampling_times, ))

    for i in range(sampling_times):
        sampled_weight = np.random.beta(beta_distribution_alpha, beta_distribution_beta)
        sampled_weight_ = 1 - sampled_weight
        mixed_weight_[i] = np.max(np.array([sampled_weight, sampled_weight_]))

    sampled = mixed_weight_
    return np.mean(sampled)


def mixmatch(model1, model2, w_b, labeled_loader, unlabeled_loader, device, param, semi_loss, sigma, optim, epoch, model_number, rtds_current_epoch):
    with torch.no_grad():

        wb = np.array(w_b)

        pb = np.zeros(shape=(len(labeled_loader.dataset), 10))
        labeled_pseudo_y_np = np.zeros(shape=(len(labeled_loader.dataset), ), dtype=np.int32)
        labeled_data = []
        for i in range(param.aug_times):
            labeled_data.append(np.zeros(shape=(len(labeled_loader.dataset), 3, 32, 32)))

        for idx, (x, y, _, _) in enumerate(labeled_loader):
            for j in range(param.aug_times):
                x_j = x[j]
                x_j_np = x_j.detach().cpu().numpy()
                labeled_data[j][idx * 100: idx * 100 + x_j_np.shape[0]] = x_j_np

            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)

            x = x.to(device)
            y = y.to(device)

            out = model1(x)
            out = torch.softmax(out, dim=1)
            out_np = out.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            s_ = 0
            y_ = 0
            bs_size = out_np.shape[0] // param.aug_times
            for i in range(param.aug_times):
                s_i = out_np[i * bs_size: (i + 1) * bs_size]
                y_i = y_np[i * bs_size: (i + 1) * bs_size]
                s_ += s_i
                y_ += y_i
            pb_bs = s_ / param.aug_times
            y_bs = np.array(y_ / param.aug_times, dtype=np.int32)

            pb[idx * 100: idx * 100 + pb_bs.shape[0]] = pb_bs
            labeled_pseudo_y_np[idx * 100: idx * 100 + pb_bs.shape[0]] = y_bs

        pb = torch.tensor(pb)
        wb = np.reshape(wb, newshape=(-1, 1))
        wb = torch.tensor(wb)
        labeled_pseudo_y_tensor = torch.tensor(labeled_pseudo_y_np).long()
        labeled_pseudo_y_tensor = torch.nn.functional.one_hot(labeled_pseudo_y_tensor, num_classes=10)
        labeled_pseudo_y_tensor = labeled_pseudo_y_tensor.float()
        pb = wb * labeled_pseudo_y_tensor + (1 - wb) * pb
        pb_sharpen = pb ** (1 / 0.5)
        #pb_sharpen = torch.tensor(pb_sharpen)
        target_labeled = pb_sharpen / pb_sharpen.sum(dim=1, keepdim=True)
        target_labeled = target_labeled.detach()

        qu = np.zeros(shape=(len(unlabeled_loader.dataset), 10))
        unlabeled_data = []
        for i in range(param.aug_times):
            unlabeled_data.append(np.zeros(shape=(len(unlabeled_loader.dataset), 3, 32, 32)))

        for idx, (x, y, _, _) in enumerate(unlabeled_loader):
            for j in range(param.aug_times):
                x_j = x[j]
                x_j_np = x_j.detach().cpu().numpy()
                unlabeled_data[j][idx * 100: idx * 100 + x_j_np.shape[0]] = x_j_np

            x = torch.cat(x, dim=0)
            y = torch.cat(y, dim=0)

            x = x.to(device)
            y = y.to(device)

            out1 = model1(x)
            out1 = torch.softmax(out1, dim=1)
            out2 = model2(x)
            out2 = torch.softmax(out2, dim=1)
            out_np1 = out1.detach().cpu().numpy()
            out_np2 = out2.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()

            out_np = (out_np1 + out_np2) / 2

            u_ = 0
            bs_size = out_np.shape[0] // param.aug_times
            for i in range(param.aug_times):
                u_i = out_np[i * bs_size: (i + 1) * bs_size]
                u_ += u_i
            qu_bs = u_ / param.aug_times

            qu[idx * 100: idx * 100 + qu_bs.shape[0]] = qu_bs

        qu = torch.tensor(qu)
        qu_sharpen = qu ** (1 / 0.5)

        Lkkk = nn.MSELoss()

        target_unlabeled = qu_sharpen / qu_sharpen.sum(dim=1, keepdim=True)
        target_unlabeled = target_unlabeled.detach()


        mix_lamb = sample_beta_distribution()

        labeled_data = np.concatenate(labeled_data, axis=0)
        unlabeled_data = np.concatenate(unlabeled_data, axis=0)
        aug_data = np.concatenate([labeled_data, unlabeled_data], axis=0)
        aug_data = torch.tensor(aug_data).float()

        aug_labeled_target = target_labeled.repeat(param.aug_times, 1)
        aug_unlabeled_target = target_unlabeled.repeat(param.aug_times, 1)
        aug_target = torch.cat([aug_labeled_target, aug_unlabeled_target], dim=0)

        randim_idx = torch.randperm(aug_data.size(0))

        input_a, input_b = aug_data, aug_data[randim_idx]
        target_a, target_b = aug_target, aug_target[randim_idx]

        mixed_input = mix_lamb * input_a + (1 - mix_lamb) * input_b
        mixed_target = mix_lamb * target_a + (1 - mix_lamb) * target_b

        labeled_data_num = len(labeled_loader.dataset) * param.aug_times

        mixed_input0 = mixed_input[:labeled_data_num]
        mixed_target0 = mixed_target[:labeled_data_num]

        mixed_input1 = mixed_input[labeled_data_num:]
        mixed_target1 = mixed_target[labeled_data_num:]

        sub_train_set0 = TensorDataset(mixed_input0, mixed_target0)
        sub_train_set1 = TensorDataset(mixed_input1, mixed_target1)
        sub_train_loader0 = DataLoader(dataset=sub_train_set0, batch_size=512, shuffle=True, drop_last=True)
        sub_train_loader1 = DataLoader(dataset=sub_train_set1, batch_size=512, shuffle=True, drop_last=True)

        test_trans = get_test_trans()
        ground_truth_fixed = np.load('./data/ground_truth_fixed.npy')
        test_dataset = sep_dataset_cifar10.cifar10_for_test(root='./data', transform=test_trans, fixed_labels=ground_truth_fixed)
        test_loader = DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=16, drop_last=False)

    L_ = 0.
    model1.train()
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

        x1 = x1.float()
        x2 = x2.float()
        y1 = y1.float()
        y2 = y2.float()

        y1_ = torch.argmax(y1, dim=1)

        logits = model1(x1)
        L_x = semi_loss(logits, y1_.long())

        logits2 = model1(x2)
        L_u = Lkkk(logits2, y2)
        L0 = L_x + sigma * L_u

        prior = torch.ones(10) / param.aug_times
        prior = prior.to(device)
        pred_mean = torch.softmax(logits, dim=1).mean(0)
        penalty = torch.sum(prior * torch.log(prior / pred_mean))

        loss = L0 + penalty

        optim.zero_grad()
        loss.backward()
        optim.step()

        L_ += loss.item()

    pre_matrix = np.zeros(shape=(50000, 10))
    raw_label = np.zeros(shape=(50000, ), dtype=np.int32)
    model1.eval()
    for idx, (x, y) in enumerate(test_loader):
        x = x.to(device)
        y = y.to(device)

        y_np = y.detach().cpu().numpy()
        raw_label[100 * idx: 100 * (idx + 1)] = y_np

        pred1 = model1(x)
        pred1 = torch.softmax(pred1, dim=1)
        pred_np1 = pred1.detach().cpu().numpy()

        pre_matrix[100 * idx: 100 * (idx + 1)] = pred_np1

    return model1, pre_matrix


class GaussianBlur:
    def __init__(self, kernel_size, min=0.1, max=2.0):
        self.min = min
        self.max = max
        self.kernel_size = kernel_size

    def __call__(self, sample):
        sample = np.array(sample)
        prob = np.random.random_sample()
        if prob < 0.5:
            sigma = (self.max - self.min) * np.random.random_sample() + self.min
            sample = cv2.GaussianBlur(sample, (self.kernel_size, self.kernel_size), sigma)
        return sample


def get_train_trans():
    train_trans = transforms.Compose([transforms.RandomResizedCrop(32),
                                      transforms.RandomHorizontalFlip(p=0.5),
                                      transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                                      transforms.RandomGrayscale(p=0.2),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return train_trans


def get_test_trans():
    test_trans = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize([0.4914, 0.4822, 0.4465], [0.2023, 0.1994, 0.2010])])
    return test_trans