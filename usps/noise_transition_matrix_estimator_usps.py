import itertools
from rtds_train_usps import *
import torch.nn as nn
import membership_selector_usps



def matrix_process(noisy_transition_matrix):
    noisy_transition_matrix_without_diag = np.zeros(shape=(noisy_transition_matrix.shape[0], noisy_transition_matrix.shape[0] - 1))
    for i in range(noisy_transition_matrix.shape[0]):
        noisy_transition_matrix_i = noisy_transition_matrix[i]
        noisy_transition_matrix_i = np.delete(noisy_transition_matrix_i, i)
        noisy_transition_matrix_without_diag[i, :] = noisy_transition_matrix_i
    return noisy_transition_matrix_without_diag


def softmax(input_vector):
    exp_x = np.exp(input_vector - np.max(input_vector))
    return exp_x / np.sum(exp_x)


def generate_replacements(vector, replacement_value, max_replacements, class_number):
    all_possibilities = []
    for num_replacements in range(1, max_replacements + 1):
        replace = (replacement_value - (class_number - num_replacements)) / num_replacements
        for positions in itertools.combinations(range(len(vector)), num_replacements):
            new_vector = vector.copy()
            for pos in positions:
                new_vector[pos] = replace
            all_possibilities.append(new_vector)
    return all_possibilities


def cross_entropy(p, q):
    q = np.clip(q, 1e-10, 1.0)
    return -np.sum(p * np.log(q))


def decide(noisy_transition_matrix):
    noisy_transition_matrix_without_diag = matrix_process(noisy_transition_matrix=noisy_transition_matrix)
    detect = []
    for i in range(noisy_transition_matrix_without_diag.shape[0]):
        prob_ = noisy_transition_matrix_without_diag[i] * 100
        prob_sum = np.sum(prob_)
        prob_ = softmax(input_vector=prob_)

        vector = [1] * int(noisy_transition_matrix.shape[0] - 1)
        max_replacements = len(vector)

        results = generate_replacements(vector=vector, replacement_value=prob_sum, max_replacements=max_replacements,
                                        class_number=int(noisy_transition_matrix.shape[0] - 1))
        result_list = [list(result) for result in results]

        ce_value = []
        for j in range(len(result_list)):
            standard_vector = np.array(result_list[j], dtype=np.float32)
            standard_distribution = softmax(standard_vector)
            ce_value.append(cross_entropy(p=standard_distribution, q=prob_))

        ce_value = np.array(ce_value)
        most_match_distribution = np.argmin(ce_value)
        if most_match_distribution == ce_value.shape[0]:
            misassignments_distribution_mode = 1
            detect.append(misassignments_distribution_mode)

        else:
            misassignments_distribution_mode = 0
            detect.append(misassignments_distribution_mode)

    if 1 in detect:
        delta_value = 1
    else:
        delta_value = 0
    return delta_value


def ntm_estimator(membership, eta, aug_times, aug_data, device, model):
    ntm_select_idx, ntm_select_pseudo_label = membership_selector_usps.membership_selector(membership=membership, eta=eta * 0.02)

    aligned_select_pseudo_labels = np.zeros(shape=(ntm_select_pseudo_label.shape[0] * aug_times,), dtype=np.int32)
    aligned_select_idx = np.zeros(shape=(ntm_select_idx.shape[0] * aug_times), dtype=np.int32)
    for i in range(ntm_select_idx.shape[0]):
        aligned_index_start = ntm_select_idx[i] * aug_times
        aligned_select_idx[i * aug_times: (i + 1) * aug_times] = np.arange(aligned_index_start, aligned_index_start + aug_times)
        aligned_select_pseudo_labels[i * aug_times: (i + 1) * aug_times] = ntm_select_pseudo_label[i]

    select_augmented_data = aug_data[aligned_select_idx]
    select_augmented_data_tensor = torch.tensor(select_augmented_data)
    aligned_select_pseudo_labels_tensor = torch.tensor(aligned_select_pseudo_labels)

    estimate_dataset = TensorDataset(select_augmented_data_tensor, aligned_select_pseudo_labels_tensor)
    estimate_loader = DataLoader(dataset=estimate_dataset, batch_size=10000, shuffle=False, drop_last=False, num_workers=16)

    pred = np.zeros(shape=(len(estimate_dataset) * aug_times, 10))
    for idx, (x, _) in enumerate(estimate_loader):
        x = x.to(device)
        x = x.view(-1, 256)
        x = x.float()
        x_np = x.detach().cpu().numpy()
        out = model(x)
        out_prob = torch.softmax(out, dim=1)
        out_np = out_prob.detach().cpu().numpy()
        if x_np.shape[0] < 10000:
            pred[idx * 10000: idx * 10000 + x_np.shape[0]] = out_np
        else:
            pred[idx * 10000: (idx + 1) * 10000] = out_np

    pred_hard = np.argmax(pred, axis=1)

    noisy_transition_matrix = np.zeros(shape=(10, 10))
    for i in range(10):
        idx_i = np.where(pred_hard == i)[0]
        pred_soft_i = pred[idx_i]
        pred_mean = np.mean(pred_soft_i, axis=0)
        noisy_transition_matrix[:, i] = pred_mean

    return noisy_transition_matrix


def sigma_estimator_hand(membership, eta, aug_times, aug_data, device, model1, model2):
    noisy_transition_matrix_0 = ntm_estimator(membership=membership, eta=eta, aug_times=aug_times, aug_data=aug_data, device=device, model=model1)
    noisy_transition_matrix_1 = ntm_estimator(membership=membership, eta=eta, aug_times=aug_times, aug_data=aug_data, device=device, model=model2)
    average_noisy_transition_matrix = (noisy_transition_matrix_0 + noisy_transition_matrix_1) / 2
    sigma = decide(noisy_transition_matrix=average_noisy_transition_matrix)
    return sigma


def ntm_process(model1, model2, warm_up_epoch, dual_model_train_loader, device):
    CE_Loss = nn.CrossEntropyLoss()
    Conf_Penalty = NegEntropy()

    optim1 = torch.optim.SGD(params=model1.parameters(), lr=0.01, momentum=0.9)
    optim2 = torch.optim.SGD(params=model2.parameters(), lr=0.01, momentum=0.9)

    model1 = warm_up(model=model1, optim=optim1, epoch=warm_up_epoch, loss1=CE_Loss, loss2=Conf_Penalty,
                     dataloader=dual_model_train_loader, device=device, crit=0.0)

    model2 = warm_up(model=model2, optim=optim2, epoch=warm_up_epoch, loss1=CE_Loss, loss2=Conf_Penalty,
                     dataloader=dual_model_train_loader, device=device, crit=0.0)

    return model1, model2




