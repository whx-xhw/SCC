import itertools
import membership_selector_cifar10
from rtds_train_cifar10 import *
import sep_dataset_cifar10


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
        sigma_value = 1
    else:
        sigma_value = 0
    return sigma_value


def ntm_estimator(membership, eta, device, model, param):
    ntm_select_idx, ntm_select_pseudo_label = membership_selector_cifar10.membership_selector(membership=membership, eta=int(eta * 1300 * 0.02))

    train_trans = get_train_trans()
    estimate_dataset = sep_dataset_cifar10.cifar10_for_ntm(root='./data/', transform=train_trans, pseudo_labels=ntm_select_pseudo_label, select_index=ntm_select_idx, aug_times=param.aug_times)
    estimate_loader = DataLoader(dataset=estimate_dataset, batch_size=100, shuffle=False, drop_last=False, num_workers=16)

    pred = np.zeros(shape=(len(estimate_dataset) * param.aug_times, 10))
    for idx, (x, y) in enumerate(estimate_loader):
        x = torch.cat(x, dim=0)
        x = x.to(device)
        x = x.float()

        out = model(x)
        out_prob = torch.softmax(out, dim=1)
        out_np = out_prob.detach().cpu().numpy()

        pred[idx * 100 * param.aug_times: idx * 100 * param.aug_times + out_np.shape[0]] = out_np

    pred_hard = np.argmax(pred, axis=1)

    noisy_transition_matrix = np.zeros(shape=(10, 10))
    for i in range(10):
        idx_i = np.where(pred_hard == i)[0]
        pred_soft_i = pred[idx_i]
        pred_mean = np.mean(pred_soft_i, axis=0)
        noisy_transition_matrix[:, i] = pred_mean

    return noisy_transition_matrix


def sigma_estimator(membership, eta, device, model1, model2, param):
    noisy_transition_matrix_0 = ntm_estimator(membership=membership, eta=eta, device=device, model=model1, param=param)
    noisy_transition_matrix_1 = ntm_estimator(membership=membership, eta=eta, device=device, model=model2, param=param)
    average_noisy_transition_matrix = (noisy_transition_matrix_0 + noisy_transition_matrix_1) / 2
    sigma = decide(noisy_transition_matrix=average_noisy_transition_matrix)
    return sigma


def ntm_process(model1, model2, warm_up_epoch, dual_model_train_loader, device):
    CE_Loss = nn.CrossEntropyLoss()
    Conf_Penalty = NegEntropy()

    optim1 = torch.optim.SGD(params=model1.parameters(), lr=0.1)
    optim2 = torch.optim.SGD(params=model2.parameters(), lr=0.1)

    model1 = warm_up_ntm(model=model1, optim=optim1, epoch=warm_up_epoch, loss1=CE_Loss,
                         loss2=Conf_Penalty, dataloader=dual_model_train_loader, device=device, crit=0.0)

    model2 = warm_up_ntm(model=model2, optim=optim2, epoch=warm_up_epoch, loss1=CE_Loss,
                         loss2=Conf_Penalty, dataloader=dual_model_train_loader, device=device, crit=0.0)

    return model1, model2