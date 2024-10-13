import numpy as np


# MS For imbalance dataset
def membership_selector(membership, eta):
    selected_idx = []
    selected_pseudo_label = []
    membership_ = np.argmax(membership, axis=1)

    for i in range(10):
        idx = np.where(membership_ == i)[0]
        membership_i_class = membership[idx, i]
        number_i = idx.shape[0]
        s_num = int(eta * number_i)
        top_eta_idx = membership_i_class.argsort()[::-1][0:s_num]
        selected_idx.append(idx[top_eta_idx])
        selected_pseudo_label.append(np.ones(shape=(s_num, ), dtype=np.int32) * i)

    selected_idx = np.concatenate(selected_idx, axis=0)
    selected_pseudo_label = np.concatenate(selected_pseudo_label, axis=0)
    return selected_idx, selected_pseudo_label