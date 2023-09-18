'''

Date: 2023-04-05 16:34:55
LastEditTime: 2023-06-17 10:16:55

Description: 
FilePath: /openset_anomaly_detection/pad_losses/energe_loss.py
have a nice day
'''


from IPython import embed
import torch


import torch.nn.functional as F

def smooth(arr, lamda1):
    new_array = arr
    copy_of_arr = 1 * arr

    arr2 = torch.zeros_like(arr)
    arr2[:, :-1, :] = arr[:, 1:, :]
    arr2[:, -1, :] = arr[:, -1, :]

    new_array2 = torch.zeros_like(new_array)
    new_array2[:, :, :-1] = new_array[:, :, 1:]
    new_array2[:, :, -1] = new_array[:, :, -1]

    # added the third direction for 3D points
    arr_3 = torch.zeros_like(copy_of_arr)
    arr_3[:, :, :, :-1] = copy_of_arr[:, :, :, 1:]
    arr_3[:, :, :, -1] = copy_of_arr[:, :, :, -1]

    loss = (torch.sum((arr2 - arr) ** 2) + torch.sum((new_array2 - new_array) ** 2) + torch.sum((arr_3 - copy_of_arr) ** 2)) / 3
    return lamda1 * loss

# TODO: Should it be calculated one by one for each point cloud in the batch?
def sparsity(arr, lamda2):
    loss = torch.mean(torch.norm(arr, dim=0))
    return lamda2 * loss



""""
ensure there are gap  between common class and noval class. 
more specifically, the noval class should has large energe while the common classes has low energy
"""
def energy_loss(logits, targets,ood_ind=5):
    # ood_ind = 5
    void_ind = 0
    T = 1.
    m_in = -12
    m_out = -6
    # Exclude class 0 from energy computation
    #!+====================================
    """"
    why not process the logits[:,ood_ind]
    """
    
    in_distribution_logits = torch.hstack([logits[:, 1:ood_ind],  logits[:, ood_ind+1:]])
    #!+====================================

    energy = -(T * torch.logsumexp(in_distribution_logits / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0:#* not OOD sample in the label 
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        loss += 0.5 * (torch.pow(F.relu(Ec_in - m_in), 2).mean() + torch.pow(F.relu(m_out - Ec_out), 2).mean())
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy


""""

"""
def dynamic_energy_loss(logits, targets,ood_ind=5,details_targets=None,m_out_max = 0,resized_point_label = 20):
    shapenet_object_point_label = resized_point_label+1

    # ood_ind = 5
    void_ind = 0
    T = 1.
    m_in = -12
    m_out = -6
    # Exclude class 0 from energy computation
    #!+====================================
    """"
    why not process the logits[:,ood_ind]
    """
    in_distribution_logits = torch.hstack([logits[:, 1:ood_ind],  logits[:, ood_ind+1:]])
    #!+====================================

    energy = -(T * torch.logsumexp(in_distribution_logits / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0: #* not OOD sample in the label 
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        
        #* in distribution  energy  and the resized point energy (20)
        tmp =  torch.pow(F.relu(Ec_in - m_in), 2).mean()  
        if (details_targets == resized_point_label).sum() != 0 :
            tmp += torch.pow(F.relu(m_out - energy[details_targets == resized_point_label]), 2).mean()

        #* out  of distribution  energy 
        all_instance_label = details_targets.unique()
        spn_labels = all_instance_label[all_instance_label>= shapenet_object_point_label]
        
        
        for spn_label in spn_labels:   
            current_instance_energies =  energy[details_targets==spn_label]
            
            if current_instance_energies.size()[0] != 0 :
                #* calculate the dynamic m_out
                dmout = m_out +  (m_out_max - m_out) * ((spn_label - shapenet_object_point_label).float() / 100)
                #* calculate the energy loss for ood
                tmp += torch.pow(F.relu(dmout - current_instance_energies), 2).mean()

        loss += tmp*0.5
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy




""""

"""
def crude_dynamic_energy_loss(logits, targets,ood_ind=5,details_targets=None,m_out_max = 0,resized_point_label = 20, resize_m_out = -6 ):
    shapenet_object_point_label = resized_point_label+1

    # ood_ind = 5
    void_ind = 0
    T = 1.
    m_in = -12
    # Exclude class 0 from energy computation
    #!+====================================
    """"
    why not process the logits[:,ood_ind]
    """
    in_distribution_logits = torch.hstack([logits[:, 1:ood_ind],  logits[:, ood_ind+1:]])
    #!+====================================

    energy = -(T * torch.logsumexp(in_distribution_logits / T, dim=1))
    Ec_out = energy[targets == ood_ind]
    Ec_in = energy[(targets != ood_ind) & (targets != void_ind)]

    loss = torch.tensor(0.).cuda()
    if Ec_out.size()[0] == 0: #* not OOD sample in the label 
        loss += torch.pow(F.relu(Ec_in - m_in), 2).mean()
    else:
        
        #* in distribution  energy  and the resized point energy (20)
        in_ebm_loss =  torch.pow(F.relu(Ec_in - m_in), 2).mean()  
        out_ebm_loss = torch.tensor(0.).cuda()
        cnt = 0
        if (details_targets == resized_point_label).sum() != 0 :
            out_ebm_loss += torch.pow(F.relu(resize_m_out - energy[details_targets == resized_point_label]), 2).mean()
            cnt += 1


        if (details_targets >= shapenet_object_point_label).sum() !=0:

            shapenet_point_energy = energy[details_targets >= shapenet_object_point_label]
            out_ebm_loss += torch.pow(F.relu(m_out_max - shapenet_point_energy), 2).mean()
            cnt += 1
            

        loss += (out_ebm_loss / (cnt + 1e-8) + in_ebm_loss)*0.5
        loss += sparsity(Ec_out, 5e-4)

    loss += smooth(energy, 3e-6)

    return loss, energy