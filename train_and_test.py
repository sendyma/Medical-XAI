import time
import torch
import torch.nn.functional as F
import numpy as np
from utils_model import eval_metrics
from utils_model.helpers import list_of_distances, make_one_hot
import random
import copy
import torchvision.transforms as transforms
from torchvision.transforms import functional as transF
import wandb


def bootstrap_sampling(prob, gt, n_times=2000):
    """Do bootstrap sampling to get confidence interval

    Args:
        gt (np_array):   gt label {0,1}
        prob (np_array): prediction in probability value [0,1]
        pred (np_array): prediction in actual label {0,1}
        n_times (int, optional): Number of resampling times. Defaults to 1000.

    Returns:
        ACC - mean, lower bound (5% quantile), upper bound (95% quantile)
        AUC, Precision, Recall, TPR, TNR, FPR, FNR
    """
    gt = np.asarray(gt)
    prob = np.asarray(prob)

    # Initialise variables
    sample_size = len(gt)
    index_array = np.arange(sample_size)
    LB_index = int(0.025 * n_times)
    UB_index = int(0.975 * n_times)
    AUC_arr = np.zeros([n_times, 3])
    AUPRC_arr = np.zeros([n_times, 3])

    # Do sampling and calculate metrics
    for i in range(n_times):
        sampled_index = np.random.choice(index_array, sample_size, replace=True)
        gt_sampled = gt[sampled_index]
        prob_sampled = prob[sampled_index]

        roc_auc_exp, pr_auc_exp, se_exp, sp_exp, acc_exp = eval_metrics.compute_roc(prob_sampled[:, 0], gt_sampled)
        roc_auc_glo, pr_auc_glo, se_glo, sp_glo, acc_glo = eval_metrics.compute_roc(prob_sampled[:, 1], gt_sampled)
        roc_auc_ens, pr_auc_ens, se_ens, sp_ens, acc_ens = eval_metrics.compute_roc(prob_sampled[:, 2], gt_sampled)

        AUC_arr[i] = np.array([roc_auc_exp, roc_auc_glo, roc_auc_ens])
        AUPRC_arr[i] = np.array([pr_auc_exp, pr_auc_glo, pr_auc_ens])
        # print(roc_auc_ens)

    # Return result dictionary
    AUC_result = {'mean': 100*AUC_arr.mean(axis=0), 'low': 100*np.sort(AUC_arr, axis=0)[LB_index], 'high': 100*np.sort(AUC_arr, axis=0)[UB_index]}
    AUPRC_result = {'mean': 100*AUPRC_arr.mean(axis=0), 'low': 100*np.sort(AUPRC_arr, axis=0)[LB_index], 'high': 100*np.sort(AUPRC_arr, axis=0)[UB_index]}
    AUC_STD = np.std(AUC_arr, axis=0)
    AUPRC_STD = np.std(AUPRC_arr, axis=0)
    AUC_result['std'] = AUC_STD
    AUPRC_result['std'] = AUPRC_STD

    return AUC_result


def softmax_KL_loss(input_logits, target_logits):
    """Takes softmax on both sides and returns KL divergence

    Note:
    - Returns the sum over all examples. Divide by the batch size afterwards
      if you want the mean.
    - Sends gradients to inputs but not the targets.
    """
    assert input_logits.size() == target_logits.size()
    input_log_softmax = F.log_softmax(input_logits, dim=1)
    target_softmax = F.softmax(target_logits, dim=1)

    loss_ce = F.kl_div(input_log_softmax, target_softmax)  # (target_softmax * (torch.log(target_softmax) - input_log_softmax)).mean()
    # loss_ce = torch.mean(-(target_softmax * input_log_softmax).sum(dim=1))
    return loss_ce


def affine_loss_fn(model, similarity_maps_old, image_ori, trans_affine_param):

    input_size = torch.Size([1536, 768])
    feat_shape = similarity_maps_old.shape[2:]
    img_shape = image_ori.shape[2:]
    ratio = int(img_shape[0] / feat_shape[0])
    batch_size = similarity_maps_old.shape[0]
    num_similarity_map = similarity_maps_old.shape[1]

    with torch.no_grad():
        output_ori, _, similarity_maps_ori, _, _, _ = model(image_ori, return_full=True)

    # similarity_maps_upsampled = torch.nn.functional.upsample(similarity_maps_ori.detach(), size=image_ori.shape[2:], mode='bilinear')
    # similarity_maps_upsampled_old = torch.nn.functional.upsample(similarity_maps_old, size=image_ori.shape[2:], mode='bilinear')

    # plt.imshow(image_ori[0, 0, :, :].cpu().numpy(), 'gray')
    # plt.show()
    #
    # plt.imshow(similarity_maps_upsampled[0, 301, :, :].cpu().numpy()*255, 'gray')
    # plt.show()
    #
    # plt.imshow(image_affine[0, 0, :, :].cpu().numpy(), 'gray')
    # plt.show()
    #
    # plt.imshow(similarity_maps_upsampled_old[0, 301, :, :].cpu().numpy()*255, 'gray')
    # plt.show()

    affine_cost = 0.0
    num_k = 1
    for b in range(batch_size):
        affine_param_b = trans_affine_param[b].tolist()
        affine_param_b[0] = affine_param_b[0][0]
        affine_param_b[2] = affine_param_b[2][0]
        affine_param_b[1] = tuple(affine_param_b[1])
        affine_param_b[3] = tuple(affine_param_b[3])
        affine_param_b = tuple(affine_param_b)
        index_neg = list(np.random.choice(range(num_similarity_map//2), num_k))
        index_pos = list(np.random.choice(range(num_similarity_map//2, num_similarity_map), num_k))
        index_all = index_neg + index_pos
        for i in index_all:
            similarity_maps_upsampled = torch.nn.functional.upsample(similarity_maps_ori[b, i][None, None, :, :], size=input_size, mode='bilinear')
            similarity_map_PIL = transforms.ToPILImage()(similarity_maps_upsampled.squeeze())   # PIL change value from [0, 1] to [0, 255]
            similarity_map_affine = transF.affine(similarity_map_PIL, *affine_param_b, fill=0)  # PIL
            similarity_map_affine = transforms.ToTensor()(similarity_map_affine).squeeze().cuda()   # [1, 1, h, w]
            # similarity_map_affine_downsampled = torch.nn.functional.interpolate(similarity_map_affine, size=feat_shape, mode='bilinear').squeeze()
            # similarity_map_affine_downsampled = similarity_map_affine[0, 0, 0::ratio, 0::ratio]

            similarity_maps_upsampled_old = torch.nn.functional.upsample(similarity_maps_old[b, i][None, None, :, :], size=input_size, mode='bilinear')
            cost_mse = torch.nn.functional.mse_loss(similarity_maps_upsampled_old.squeeze(), similarity_map_affine, reduction='none')
            mask = (similarity_map_affine != 0)
            cost_one = (cost_mse * mask).sum() / (mask.sum() + 1e-10)

            affine_cost = affine_cost + cost_one

    # bbb = np.array(similarity_map_affine.cpu().numpy().squeeze())
    # plt.imshow(bbb, 'gray')
    # plt.show()
    #
    # ccc = similarity_maps_upsampled_old.detach().cpu().numpy().squeeze()
    # plt.imshow(ccc, 'gray')
    # plt.show()
    #
    # cv2.imwrite('affine.png', similarity_map_affine.cpu().numpy().squeeze() * 255)
    # cv2.imwrite('aaaa.png', similarity_maps_upsampled_old.detach().cpu().numpy().squeeze() * 255)

    return affine_cost / (batch_size * 2 * num_k), output_ori



def _pretraining_globalnet(model, dataloader, optimizer=None, class_specific=True, use_l1_mask=True, log=print, epoch=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_examples_0 = 0
    n_correct_0 = 0
    n_examples_1 = 0
    n_correct_1 = 0
    n_batches = 0

    total_cross_entropy_global = 0
    probabilities_global = []
    all_targets = []

    for i, (image, label, img_name) in enumerate(dataloader):

        assert isinstance(image, list)

        input = image[0].cuda()   # image[0]: augmented image, image[0]: original image, image[0]: augment parameters
        target = label.cuda()

        global_logits = model(model.module.forward_backbone(input))

        global_prob = torch.softmax(global_logits, dim=1)

        probabilities_global.append(global_prob.detach().cpu().numpy())
        all_targets.append(label.numpy())

        cross_entropy_global = F.cross_entropy(global_logits, target)

        loss = cross_entropy_global

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # evaluation statistics
        _, predicted = torch.max(global_prob.data, 1)
        n_examples += target.size(0)
        n_correct += (predicted == target).sum().item()

        n_correct_0 += ((predicted == target) * (target == 0)).sum().item()
        n_examples_0 += (target == 0).sum().item()

        n_correct_1 += ((predicted == target) * (target == 1)).sum().item()
        n_examples_1 += (target == 1).sum().item()

        n_batches += 1

        if i % 20 == 0:
            print(
                '{} {} \tLoss_total: {:.4f} \tLoss_CE_global: {:.4f} \tAcc_0: {:.4f}  \tAcc_1: {:.4f}'.format(
                    i, len(dataloader), loss.item(), cross_entropy_global.item(),
                    n_correct_0 / (n_examples_0 + 1e-8) * 100, n_correct_1 / (n_examples_1 + 1e-8) * 100))

            wandb.log({
                "Train Total Loss": loss.item(),
                "Train Global CE Loss": cross_entropy_global.item(),
            })

    end = time.time()

    probabilities_global = np.concatenate(probabilities_global, axis=0)

    all_targets = np.concatenate(all_targets, axis=0)

    roc_auc_glo, pr_auc_glo, se_glo, sp_glo, acc_glo = eval_metrics.compute_roc(probabilities_global[:, 1], all_targets)  # only need prob score of positive class

    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent global: \t{0}'.format(total_cross_entropy_global / n_batches))

    log('\tpr-auc_global: \t\t{0}%'.format(pr_auc_glo * 100))
    log('\tauc_global: \t\t{0}%'.format(roc_auc_glo * 100))

    return roc_auc_glo


def _testing_globalnet(model, dataloader, class_specific=True, log=print, epoch=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_examples_0 = 0
    n_correct_0 = 0
    n_examples_1 = 0
    n_correct_1 = 0
    n_batches = 0

    total_cross_entropy_global = 0
    probabilities_global = []
    all_targets = []

    for i, (image, label, img_name) in enumerate(dataloader):

        input = image.cuda()
        target = label.cuda()

        grad_req = torch.no_grad()
        with grad_req:

            global_logits = model(model.module.forward_backbone(input))

            global_prob = torch.softmax(global_logits, dim=1)

            probabilities_global.append(global_prob.detach().cpu().numpy())
            all_targets.append(label.numpy())

            cross_entropy_global = F.cross_entropy(global_logits, target)

            # evaluation statistics
            _, predicted = torch.max(global_prob.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_correct_0 += ((predicted == target) * (target == 0)).sum().item()    # specificity
            n_examples_0 += (target == 0).sum().item()

            n_correct_1 += ((predicted == target) * (target == 1)).sum().item()    # sensitivity
            n_examples_1 += (target == 1).sum().item()

            n_batches += 1
            total_cross_entropy_global += cross_entropy_global.item()

    end = time.time()

    probabilities_global = np.concatenate(probabilities_global, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)

    roc_auc_glo, pr_auc_glo, se_glo, sp_glo, acc_glo = eval_metrics.compute_roc(probabilities_global[:, 1], all_targets)  # only need prob score of positive class

    log('\ttime: \t{0}'.format(end - start))

    log('\tglobal acc: \t\t{0}%'.format(acc_glo * 100))
    log('\tglobal sen: \t\t{0}%'.format(se_glo * 100))
    log('\tglobal spe: \t\t{0}%'.format(sp_glo * 100))
    log('\tpr-auc_global: \t\t{0}%'.format(pr_auc_glo * 100))
    log('\tauc_global: \t\t{0}%'.format(roc_auc_glo * 100))

    wandb.log({
        "Test Global Loss": total_cross_entropy_global / n_batches,
        "Test Global AUC": roc_auc_glo * 100,
        "Test Global PR-AUC": pr_auc_glo * 100,
        "Test Global Spe": sp_glo * 100,
        "Test Global Sen": se_glo * 100,
        "Test Global Acc": acc_glo * 100,
    })

    # ROC-AUC Curve
    wandb.log({"ROC Global": wandb.plot.roc_curve(all_targets, probabilities_global, title="Global ROC-AUC")})
    # Precision Recall Curve
    wandb.log({"PR Global": wandb.plot.pr_curve(all_targets, probabilities_global, title="Global PR-AUC")})
    # Confusion matrix
    wandb.log({"Global conf_mat": wandb.plot.confusion_matrix(preds=probabilities_global.argmax(axis=-1), y_true=all_targets, title="Global CM")})

    return roc_auc_glo


def _validation_globalnet(model, dataloader, class_specific=True, log=print,  epoch=None, valid_loss_min=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''

    n_batches = 0
    total_cross_entropy = 0.0

    for i, (image, label, img_name) in enumerate(dataloader):

        input = image.cuda()
        target = label.cuda()

        grad_req = torch.no_grad()
        with grad_req:
            n_batches += 1
            global_logits = model(model.module.forward_backbone(input))
            cross_entropy = F.cross_entropy(global_logits, target)
            total_cross_entropy += cross_entropy.item()

    total_cross_entropy = total_cross_entropy / n_batches

    wandb.log({
        "Valid Loss Global": total_cross_entropy,
        })

    # save model if validation loss has decreased
    if total_cross_entropy <= valid_loss_min:
        return total_cross_entropy
    else:
        return valid_loss_min


def _training_protopnet(model_teacher, model_student, labeled_loader, unlabeled_loader, optimizer_teacher, optimizer_student,
                        class_specific=True, use_l1_mask=True, coefs=None, log=print, epoch=None, train_type=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer_teacher is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_examples_0 = 0
    n_correct_0 = 0
    n_examples_1 = 0
    n_correct_1 = 0

    use_affine = True
    temperature = 0.7   # 0.7
    threshold = 0.6
    lambda_u = 1.0
    uda_epochs = 2.0
    steps_per_epoch = len(unlabeled_loader)  # unlabeled samples
    uda_steps = uda_epochs * steps_per_epoch

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model_teacher.train()
    model_student.train()

    lr_s = 1e-3  # student's base lr, 1.0
    for step in range(0, steps_per_epoch):

        try:
            images_labeled, targets_labeled, img_name_labeled = labeled_iter.next()   #  labeled samples are shuffled already
        except:
            labeled_iter = iter(labeled_loader)
            images_labeled, targets_labeled, img_name_labeled = labeled_iter.next()

        try:
            (images_unlabeled_ori, images_unlabeled_aug, affine_para), targets_unlabeled, flag_islabeled, img_name_unlabeled = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabeled_loader)
            (images_unlabeled_ori, images_unlabeled_aug, affine_para), targets_unlabeled, flag_islabeled, img_name_unlabeled = unlabeled_iter.next()

        images_labeled = images_labeled.cuda()
        targets_labeled = targets_labeled.cuda()
        images_unlabeled_ori = images_unlabeled_ori.cuda()
        images_unlabeled_aug = images_unlabeled_aug.cuda()
        targets_unlabeled = targets_unlabeled.cuda()

        batch_size_labeled = images_labeled.shape[0]
        batch_size_unlabeled = images_unlabeled_aug.shape[0]

        feat_backbone_labeled = model_teacher.module.forward_backbone(images_labeled).detach()
        feat_backbone_unlabeled_aug = model_teacher.module.forward_backbone(images_unlabeled_aug).detach()
        feat_backbone_unlabeled_ori = model_teacher.module.forward_backbone(images_unlabeled_ori).detach()

        #################################################################################################  Student
        # train student model on unlabeled data
        pred_unlabeled_student, min_distances_unlabeled, similarity_maps_unlabeled = model_student(feat_backbone_unlabeled_aug)   # unlabeled aug
        pred_unlabeled_teacher = model_teacher(feat_backbone_unlabeled_aug)    # unlabeled aug

        optimizer_student.zero_grad()
        pred_unlabeled_teacher_argmax = torch.argmax(pred_unlabeled_teacher, dim=1)   # hard_pseudo unlabeled
        loss_student_unlabeled_ce = F.cross_entropy(pred_unlabeled_student, pred_unlabeled_teacher_argmax)
        # soft_pseudo_unlabeled_teacher = torch.softmax(pred_unlabeled_teacher.detach() / temperature, dim=-1)    # 0.7 temperature
        # loss_student_unlabeled_ce = torch.mean(-(soft_pseudo_unlabeled_teacher * torch.log_softmax(pred_unlabeled_student, dim=-1)).sum(dim=-1))

        # affine consistency loss
        if use_affine:
            affine_cost, pred_unlabeled_student_ori = affine_loss_fn(model_student, similarity_maps_unlabeled, feat_backbone_unlabeled_ori, affine_para)
            affine_cost = affine_cost * 200
            affine_cls_cost = softmax_KL_loss(pred_unlabeled_student, pred_unlabeled_student_ori)
        else:
            affine_cost = torch.tensor(0.0).cuda()
            affine_cls_cost = torch.tensor(0.0).cuda()

        cluster_cost = 0.0
        separation_cost = 0.0
        num_proto_per_class = model_student.module.num_neg_proto

        for b in range(batch_size_unlabeled):
            if flag_islabeled[b] == False:
                continue
            if targets_unlabeled[b] == 0:  # targets_unlabeled is equal to its ground-truth label
                cluster_cost += torch.min(min_distances_unlabeled[b][0:num_proto_per_class])
                separation_cost += torch.min(min_distances_unlabeled[b][num_proto_per_class:])
            else:
                cluster_cost += torch.min(min_distances_unlabeled[b][num_proto_per_class:])
                separation_cost += torch.min(min_distances_unlabeled[b][0:num_proto_per_class])
        cluster_cost = 0.02 * cluster_cost / (flag_islabeled.sum() + 1e-8)
        separation_cost = 0.02 * separation_cost / (flag_islabeled.sum() + 1e-8)



        # soft_pseudo_unlabeled_teacher = F.softmax(pred_unlabeled_teacher.detach(), dim=-1)
        # max_probs = soft_pseudo_unlabeled_teacher.max(dim=-1)[0]
        # mask = max_probs.ge(0.7).float()    # 0.7 with high certainty
        # nnn = 0.0
        # for b in range(batch_size_unlabeled):
        #     if flag_islabeled[b] == True:
        #         if targets_unlabeled[b] == 0:  # targets_unlabeled
        #             cluster_cost += torch.min(min_distances_unlabeled[b][0:num_proto_per_class])
        #             separation_cost += torch.min(min_distances_unlabeled[b][num_proto_per_class:])
        #         else:
        #             cluster_cost += torch.min(min_distances_unlabeled[b][num_proto_per_class:])
        #             separation_cost += torch.min(min_distances_unlabeled[b][0:num_proto_per_class])
        #         nnn = nnn + 1
        #     else:
        #         if mask[b] == 1:
        #             if pred_unlabeled_teacher_argmax[b] == 0:     # pred_unlabeled_teacher_argmax
        #                 cluster_cost += torch.min(min_distances_unlabeled[b][0:num_proto_per_class])
        #                 separation_cost += torch.min(min_distances_unlabeled[b][num_proto_per_class:])
        #             else:
        #                 cluster_cost += torch.min(min_distances_unlabeled[b][num_proto_per_class:])
        #                 separation_cost += torch.min(min_distances_unlabeled[b][0:num_proto_per_class])
        #             nnn = nnn + 1
        # cluster_cost = 0.02 * cluster_cost / (flag_islabeled.sum() + 1e-8)
        # separation_cost = 0.02 * separation_cost / (flag_islabeled.sum() + 1e-8)




        loss_student_unlabeled = (
                1.0 * loss_student_unlabeled_ce
                + 0.1 * cluster_cost
                + 0.1 * F.relu(10 - separation_cost)    # 10
                + 1.0 * affine_cost                     # 1.0
                # + 1.0 * affine_cls_cost                 # 1.0
                )
        loss_student_unlabeled.backward()  # grads are obtained but not update params
        grad_student_unlabeled = copy.deepcopy([para.grad for para in model_student.parameters() if para.requires_grad])
        optimizer_student.step()           # update params using a batch of unlabeled data

        # calculate student's performance on labeled data
        optimizer_student.zero_grad()
        pred_labeled_student, _, _ = model_student(feat_backbone_labeled)      # labeled
        loss_student_labeled = F.cross_entropy(pred_labeled_student, targets_labeled)
        loss_student_labeled.backward()
        grad_student_labeled = copy.deepcopy([para.grad for para in model_student.parameters() if para.requires_grad])

        grad_t0mult1 = 0.0
        for i in range(len(grad_student_unlabeled)):
            grad_t0mult1 += torch.sum(torch.mul(grad_student_unlabeled[i], grad_student_labeled[i]))  # dot product
        h = lr_s * grad_t0mult1

        #################################################################################################  Teacher
        # Compute the teacher’s gradient from the student’s feedback:
        loss_teacher_unlabeled = F.cross_entropy(pred_unlabeled_teacher, pred_unlabeled_teacher_argmax)  # unlabeled aug
        optimizer_teacher.zero_grad()
        loss_teacher_unlabeled.backward()

        # grad_teacher_unlabeled = copy.deepcopy([h * para.grad for para in model_teacher.parameters() if para.requires_grad])
        grad_teacher_unlabeled_temp = []
        for para in model_teacher.parameters():
            if para.grad is None:
                grad_teacher_unlabeled_temp.append(None)
            else:
                grad_teacher_unlabeled_temp.append(h * para.grad)
        grad_teacher_unlabeled = copy.deepcopy(grad_teacher_unlabeled_temp)

        # Compute the teacher’s supervised loss:
        optimizer_teacher.zero_grad()
        pred_labeled_teacher = model_teacher(feat_backbone_labeled)    # labeled
        loss_teacher_labeled = F.cross_entropy(pred_labeled_teacher, targets_labeled)

        # # Compute the teacher’s UDA loss (not used):
        # pred_unlabeled_teacher_ori = model_teacher(feat_backbone_unlabeled_ori)  # unlabeled ori
        # pred_unlabeled_teacher_aug = model_teacher(feat_backbone_unlabeled_aug)    # unlabeled aug
        # soft_pseudo_label = torch.softmax(pred_unlabeled_teacher_ori.detach() / temperature, dim=-1)  # 0.7
        # max_probs, hard_pseudo_label = torch.max(soft_pseudo_label, dim=-1)
        # mask = max_probs.ge(threshold).float()  # 0.6 with high certainty
        # t_loss_uda = torch.mean(-(soft_pseudo_label * torch.log_softmax(pred_unlabeled_teacher_aug, dim=-1)).sum(dim=-1) * mask)
        # weight_uda = lambda_u * min(1., (step + epoch * steps_per_epoch) / uda_steps)
        # t_loss_labeled_uda = loss_teacher_labeled + weight_uda * t_loss_uda   # uda loss teacher
        t_loss_uda = torch.tensor(0.0).cuda()
        weight_uda = 0.0
        t_loss_labeled_uda = loss_teacher_labeled
        t_loss_labeled_uda.backward()

        # grad_teacher_labeled = copy.deepcopy([para.grad for para in model_teacher.parameters() if para.requires_grad])
        grad_teacher_labeled_temp = []
        for para in model_teacher.parameters():
            if para.grad is None:
                grad_teacher_labeled_temp.append(None)
            else:
                grad_teacher_labeled_temp.append(para.grad)
        grad_teacher_labeled = copy.deepcopy(grad_teacher_labeled_temp)

        # Update the teacher:
        for grad_index, para in enumerate(model_teacher.parameters()):
            if para.requires_grad:
                para.grad = grad_teacher_unlabeled[grad_index] + grad_teacher_labeled[grad_index]   # optimizer_teacher.step() will update params
        optimizer_teacher.step()

        loss_teacher_unlabeled_show = F.cross_entropy(pred_unlabeled_teacher.detach(), targets_unlabeled)

        # avoid negative contributions of prototypes
        #####################################################################
        model_student.module.last_layer_neg.weight.data.clamp_(min=0.0)
        model_student.module.last_layer_pos.weight.data.clamp_(min=0.0)
        #####################################################################

        _, predicted = torch.max(pred_labeled_student.data, 1)
        n_correct_0 += ((predicted == targets_labeled) * (targets_labeled == 0)).sum().item()
        n_examples_0 += (targets_labeled == 0).sum().item()

        n_correct_1 += ((predicted == targets_labeled) * (targets_labeled == 1)).sum().item()
        n_examples_1 += (targets_labeled == 1).sum().item()

        if step % 20 == 0:
            print(
                '{} {} \tLoss_t: {:.4f} \tLoss_t_uda: {:.4f} \tLoss_t_l: {:.4f} \tloss_t_unl: {:.4f} '
                '\tLoss_s_l: {:.4f} \tLoss_s_unl: {:.4f} \tLoss_s_unl_ce: {:.4f} '
                '\tLoss_s_clust: {:.4f} \tLoss_s_spera: {:.4f} \tLoss_affine: {:.4f} \tLoss_Cls_affine: {:.4f}  h_factor: {:.4f} '
                '\tAcc_0: {:.4f}, Acc_1: {:.4f}'.format(
                    step, steps_per_epoch, t_loss_labeled_uda.item(), t_loss_uda.item(), loss_teacher_labeled.item(), loss_teacher_unlabeled.item(),
                    loss_student_labeled.item(), loss_student_unlabeled.item(), loss_student_unlabeled_ce.item(),
                    cluster_cost.item(), separation_cost.item(), affine_cost.item(), affine_cls_cost.item(), h,
                    n_correct_0 / (n_examples_0 + 1e-8) * 100, n_correct_1 / (n_examples_1 + 1e-8) * 100))

            wandb.log({
                "Train Teacher Labeled Loss": loss_teacher_labeled.item(),
                "Train Teacher Unlabeled Loss": loss_teacher_unlabeled.item(),
                "Train Teacher UDA Loss": t_loss_uda.item(),
                "Train Teacher Labeled UDA Loss": t_loss_labeled_uda.item(),
                "Train Teacher UDA Weight": weight_uda,
                "Train Teacher Unlabeled Show Loss": loss_teacher_unlabeled_show.item(),
                "Train Student Labeled Loss": loss_student_labeled.item(),
                "Train Student Unlabeled Loss": loss_student_unlabeled.item(),
                "Train Student Unlabeled CE Loss": loss_student_unlabeled_ce.item(),
                "Train Cluster Loss": cluster_cost.item(),
                "Train Separation Loss": separation_cost.item(),
                "Train Affine Loss": affine_cost.item(),
                "Train Affine Cls Loss": affine_cls_cost.item(),
                "Train h Factor": h,
            })

    return 0.0


def _retraining_globalnet(model_teacher, model_student, labeled_loader, unlabeled_loader, optimizer_teacher, optimizer_student,
                         class_specific=True, use_l1_mask=True, coefs=None, log=print, epoch=None, train_type=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    is_train = optimizer_teacher is not None
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_examples_0 = 0
    n_correct_0 = 0
    n_examples_1 = 0
    n_correct_1 = 0

    temperature = 0.7   # 0.7
    threshold = 0.6
    lambda_u = 1.0
    uda_epochs = 5.0
    steps_per_epoch = len(unlabeled_loader)  # unlabeled samples

    labeled_iter = iter(labeled_loader)
    unlabeled_iter = iter(unlabeled_loader)

    model_teacher.train()
    model_student.eval()

    for step in range(0, steps_per_epoch):

        try:
            images_labeled, targets_labeled, img_name_labeled = labeled_iter.next()
        except:
            labeled_iter = iter(labeled_loader)
            images_labeled, targets_labeled, img_name_labeled = labeled_iter.next()

        try:
            images_unlabeled, targets_unlabeled, flag_islabeled, img_name_unlabeled = unlabeled_iter.next()
        except:
            unlabeled_iter = iter(unlabeled_loader)
            images_unlabeled, targets_unlabeled, flag_islabeled, img_name_unlabeled = unlabeled_iter.next()
        images_unlabeled = images_unlabeled[1]  # augmented

        # try:
        #     images_unlabeled, targets_unlabeled, img_name_unlabeled = unlabeled_iter.next()
        # except:
        #     unlabeled_iter = iter(unlabeled_loader)
        #     images_unlabeled, targets_unlabeled, img_name_unlabeled = unlabeled_iter.next()

        images_labeled = images_labeled.cuda()
        targets_labeled = targets_labeled.cuda()
        images_unlabeled = images_unlabeled.cuda()
        targets_unlabeled = targets_unlabeled.cuda()

        batch_size_labeled = images_labeled.shape[0]

        feat_backbone_unlabeled = model_teacher.module.forward_backbone(images_unlabeled).detach()
        #################################################################################################  student
        # train the teacher model on unlabeled data
        pred_unlabeled_student, _, _ = model_student(feat_backbone_unlabeled)   # unlabeled
        pred_unlabeled_teacher = model_teacher(feat_backbone_unlabeled)

        optimizer_teacher.zero_grad()
        pred_unlabeled_student_argmax = torch.argmax(pred_unlabeled_student, dim=1)   # hard_pseudo unlabeled
        loss_teacher_unlabeled = F.cross_entropy(pred_unlabeled_teacher, pred_unlabeled_student_argmax)

        # soft_pseudo_unlabeled_student = torch.softmax(pred_unlabeled_student.detach() / temperature, dim=-1)    # 0.7
        # loss_teacher_unlabeled = torch.mean(-(soft_pseudo_unlabeled_student *
        #                                       torch.log_softmax(pred_unlabeled_teacher, dim=-1)).sum(dim=-1))

        loss_teacher = loss_teacher_unlabeled      # 2) GT; 3) GT + PL

        loss_teacher.backward()
        optimizer_teacher.step()

        loss_teacher_unlabeled_show = F.cross_entropy(pred_unlabeled_teacher.detach(), targets_unlabeled)

        _, predicted = torch.max(pred_unlabeled_teacher.data, 1)
        n_correct_0 += ((predicted == targets_unlabeled) * (targets_unlabeled == 0)).sum().item()
        n_examples_0 += (targets_unlabeled == 0).sum().item()

        n_correct_1 += ((predicted == targets_unlabeled) * (targets_unlabeled == 1)).sum().item()
        n_examples_1 += (targets_unlabeled == 1).sum().item()

        if step % 20 == 0:
            print(
                '{} {} \tLoss_t_l: {:.4f} \tloss_t_unl: {:.4f} \tAcc_0: {:.4f}  \tAcc_1: {:.4f}'.format(
                    step, steps_per_epoch, loss_teacher_unlabeled.item(), loss_teacher_unlabeled.item(),
                    n_correct_0 / (n_examples_0 + 1e-8) * 100, n_correct_1 / (n_examples_1 + 1e-8) * 100))

            wandb.log({
                "Train Teacher Labeled Loss": loss_teacher_unlabeled.item(),
                "Train Teacher Unlabeled Loss": loss_teacher_unlabeled.item(),
                "Train Teacher Unlabeled Show Loss": loss_teacher_unlabeled_show.item(),
            })

    return 0.0


def _testing_twobranches(model_teacher, model_student, dataloader, class_specific=True, log=print, epoch=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''
    start = time.time()
    n_examples = 0
    n_correct = 0
    n_examples_0 = 0
    n_correct_0 = 0
    n_examples_1 = 0
    n_correct_1 = 0
    n_batches = 0
    total_cross_entropy_explain = 0
    total_cross_entropy_global = 0

    total_cluster_cost = 0
    total_avg_cluster_cost = 0
    # separation cost is meaningful only for class_specific
    total_separation_cost = 0
    total_avg_separation_cost = 0

    total_diversity_cost = 0

    probabilities_explain = []
    probabilities_global = []
    probabilities_ensemble = []
    all_targets = []

    for i, (image, label, img_name) in enumerate(dataloader):

        input = image.cuda()
        target = label.cuda()

        grad_req = torch.no_grad()
        with grad_req:

            feat_backbone = model_teacher.module.forward_backbone(input).detach()
            global_logits = model_teacher(feat_backbone)
            explain_logits, min_distances, _ = model_student(feat_backbone)

            explain_prob = torch.softmax(explain_logits, dim=1)
            global_prob = torch.softmax(global_logits, dim=1)
            ensemble_prob = (explain_prob + global_prob) * 0.5

            probabilities_explain.append(explain_prob.detach().cpu().numpy())
            probabilities_global.append(global_prob.detach().cpu().numpy())
            probabilities_ensemble.append(ensemble_prob.detach().cpu().numpy())
            all_targets.append(label.numpy())

            cross_entropy_explain = F.cross_entropy(explain_logits, target)
            cross_entropy_global = F.cross_entropy(global_logits, target)

            # prototypes_of_correct_class is a tensor of shape batch_size * num_prototypes
            prototypes_of_correct_class = torch.t(model_student.module.prototype_class_identity[:, label]).cuda()
            prototypes_of_wrong_class = 1 - prototypes_of_correct_class

            avg_cluster_cost_all = []
            avg_separation_cost_all = []

            # num_proto_per_class = model.module.prototype_vectors.shape[0] // 2
            num_proto_per_class = model_student.module.num_neg_proto
            batch_size = min_distances.shape[0]

            cluster_cost = 0.0
            separation_cost = 0.0
            for b in range(batch_size):
                if label[b] == 0:
                    cluster_cost += torch.min(min_distances[b][0:num_proto_per_class])
                    separation_cost += torch.min(min_distances[b][num_proto_per_class:])
                else:
                    cluster_cost += torch.min(min_distances[b][num_proto_per_class:])
                    separation_cost += torch.min(min_distances[b][0:num_proto_per_class])
            cluster_cost = 0.02 * cluster_cost / batch_size            ################################
            separation_cost = 0.02 * separation_cost / batch_size      ################################

            # calculate avg cluster and separation cost
            avg_cluster_cost = (torch.sum(min_distances * prototypes_of_correct_class, dim=1) / torch.sum(prototypes_of_correct_class, dim=1)).mean()
            avg_cluster_cost_all.append(avg_cluster_cost)
            avg_separation_cost = (torch.sum(min_distances * prototypes_of_wrong_class, dim=1) / torch.sum(prototypes_of_wrong_class, dim=1)).mean()
            avg_separation_cost_all.append(avg_separation_cost)

            # evaluation statistics
            _, predicted = torch.max(ensemble_prob.data, 1)
            n_examples += target.size(0)
            n_correct += (predicted == target).sum().item()

            n_correct_0 += ((predicted == target) * (target == 0)).sum().item()    # specificity
            n_examples_0 += (target == 0).sum().item()

            n_correct_1 += ((predicted == target) * (target == 1)).sum().item()    # sensitivity
            n_examples_1 += (target == 1).sum().item()

            n_batches += 1
            total_cross_entropy_explain += cross_entropy_explain.item()
            total_cross_entropy_global += cross_entropy_global.item()
            total_cluster_cost += cluster_cost.item()
            total_separation_cost += separation_cost.item()
            total_avg_separation_cost += avg_separation_cost.item()
            total_avg_cluster_cost += avg_cluster_cost.item()

    end = time.time()

    probabilities_explain = np.concatenate(probabilities_explain, axis=0)
    probabilities_global = np.concatenate(probabilities_global, axis=0)
    probabilities_ensemble = np.concatenate(probabilities_ensemble, axis=0)

    all_targets = np.concatenate(all_targets, axis=0)

    roc_auc_exp, pr_auc_exp, se_exp, sp_exp, acc_exp = eval_metrics.compute_roc(probabilities_explain[:, 1], all_targets)   # only need prob score of positive class
    roc_auc_glo, pr_auc_glo, se_glo, sp_glo, acc_glo = eval_metrics.compute_roc(probabilities_global[:, 1], all_targets)  # only need prob score of positive class
    roc_auc_ens, pr_auc_ens, se_ens, sp_ens, acc_ens = eval_metrics.compute_roc(probabilities_ensemble[:, 1], all_targets)  # only need prob score of positive class

    # bootstrap_eval = True
    bootstrap_eval = False
    if bootstrap_eval:
        prob = np.stack((probabilities_explain[:, 1], probabilities_global[:, 1], probabilities_ensemble[:, 1]), axis=1)
        bootstrap_AUC = bootstrap_sampling(prob, all_targets)
        print(bootstrap_AUC)

    log('\ttime: \t{0}'.format(end - start))
    log('\tcross ent explain: \t{0}'.format(total_cross_entropy_explain / n_batches))
    log('\tcross ent global: \t{0}'.format(total_cross_entropy_global / n_batches))
    log('\tcluster: \t{0}'.format(total_cluster_cost / n_batches))
    log('\tavg cluster:\t{0}'.format(total_avg_cluster_cost / n_batches))
    if class_specific:
        log('\tseparation:\t{0}'.format(total_separation_cost / n_batches))
        log('\tavg separation:\t{0}'.format(total_avg_separation_cost / n_batches))
    log('\tdiversity:\t{0}'.format(total_diversity_cost / n_batches))
    log('\texplain acc: \t\t{0}%'.format(acc_exp * 100))
    log('\texplain sen: \t\t{0}%'.format(se_exp * 100))
    log('\texplain spe: \t\t{0}%'.format(sp_exp * 100))
    log('\tpr-auc_explain: \t\t{0}%'.format(pr_auc_exp * 100))
    log('\tpr-auc_global: \t\t{0}%'.format(pr_auc_glo * 100))
    log('\tpr-auc_ensemble: \t\t{0}%'.format(pr_auc_ens * 100))
    log('\tauc_explain: \t\t{0}%'.format(roc_auc_exp * 100))
    log('\tauc_global: \t\t{0}%'.format(roc_auc_glo * 100))
    log('\tauc_ensemble: \t\t{0}%'.format(roc_auc_ens * 100))

    p = model_student.module.prototype_vectors.view(model_student.module.num_prototypes, -1).cpu()
    with torch.no_grad():
        neg_avg_pair_dist = torch.mean(list_of_distances(p[0:p.shape[0] // 2], p[0:p.shape[0] // 2]))
        pos_avg_pair_dist = torch.mean(list_of_distances(p[p.shape[0] // 2:], p[p.shape[0] // 2:]))
        all_avg_pair_dist = torch.mean(list_of_distances(p, p))
    log('\tneg dist pair: \t{0}'.format(neg_avg_pair_dist.item()))
    log('\tpos dist pair: \t{0}'.format(pos_avg_pair_dist.item()))
    log('\tall dist pair: \t{0}'.format(all_avg_pair_dist.item()))

    wandb.log({
        "Test Explain Loss": total_cross_entropy_explain / n_batches,
        "Test Global Loss": total_cross_entropy_global / n_batches,
        "Test Fusion Loss": (total_cross_entropy_explain + total_cross_entropy_global) / n_batches,
        "Test Explain AUC": roc_auc_exp * 100,
        "Test Global AUC": roc_auc_glo * 100,
        "Test Fusion AUC": roc_auc_ens * 100,
        "Test Explain PR-AUC": pr_auc_exp * 100,
        "Test Global PR-AUC": pr_auc_glo * 100,
        "Test Fusion PR-AUC": pr_auc_ens * 100,
        "Test Explain Spe": sp_exp * 100,
        "Test Explain Sen": se_exp * 100,
        "Test Explain Acc": acc_exp * 100,
    })

    # ROC-AUC Curve
    wandb.log({"ROC Explain": wandb.plot.roc_curve(all_targets, probabilities_explain, title="Explain ROC-AUC")})
    wandb.log({"ROC Global": wandb.plot.roc_curve(all_targets, probabilities_global, title="Global ROC-AUC")})
    wandb.log({"ROC Fusion": wandb.plot.roc_curve(all_targets, probabilities_ensemble, title="Fusion ROC-AUC")})
    # Precision Recall Curve
    wandb.log({"PR Explain": wandb.plot.pr_curve(all_targets, probabilities_explain, title="Explain PR-AUC")})
    wandb.log({"PR Global": wandb.plot.pr_curve(all_targets, probabilities_global, title="Global PR-AUC")})
    wandb.log({"PR Fusion": wandb.plot.pr_curve(all_targets, probabilities_ensemble, title="Fusion PR-AUC")})
    # Confusion matrix
    wandb.log({"Explain conf_mat": wandb.plot.confusion_matrix(preds=probabilities_explain.argmax(axis=-1), y_true=all_targets, title="Explain CM")})
    wandb.log({"Global conf_mat": wandb.plot.confusion_matrix(preds=probabilities_global.argmax(axis=-1), y_true=all_targets, title="Global CM")})
    wandb.log({"Fusion conf_mat": wandb.plot.confusion_matrix(preds=probabilities_ensemble.argmax(axis=-1), y_true=all_targets, title="Fusion CM")})

    return roc_auc_ens


def _validation_protopnet(model_teacher, model_student, dataloader, class_specific=True, log=print, epoch=None, valid_loss_min=None):
    '''
    model: the multi-gpu model
    dataloader:
    optimizer: if None, will be test evaluation
    '''

    n_batches = 0
    total_cross_entropy = 0.0

    for i, (image, label, img_name) in enumerate(dataloader):
        input = image.cuda()
        target = label.cuda()

        grad_req = torch.no_grad()
        with grad_req:
            n_batches += 1
            feat_backbone = model_teacher.module.forward_backbone(input).detach()
            explain_logits, min_distances, _ = model_student(feat_backbone)
            cross_entropy = F.cross_entropy(explain_logits, target)
            total_cross_entropy += cross_entropy.item()

    total_cross_entropy = total_cross_entropy / n_batches

    wandb.log({
        "Valid Loss Explain": total_cross_entropy,
    })

    # save model if validation loss has decreased
    if total_cross_entropy <= valid_loss_min:
        return total_cross_entropy
    else:
        return valid_loss_min


def train(model_teacher, model_student, labeled_loader, unlabeled_loader, optimizer_teacher, optimizer_student,
          class_specific=False, coefs=None, log=print, epoch=None, train_type=None):

    log('\ttrain')

    if train_type == 'pretraining_globalnet':
        assert (optimizer_teacher is not None)
        model_teacher.train()
        return _pretraining_globalnet(model=model_teacher, dataloader=labeled_loader, optimizer=optimizer_teacher,
                                      class_specific=class_specific, log=log, epoch=epoch)
    if train_type == 'training_protopnet':  # 1st stage
        assert (optimizer_teacher is not None) and (optimizer_student is not None)
        model_teacher.train()
        model_student.train()
        return _training_protopnet(model_teacher, model_student, labeled_loader, unlabeled_loader,
                                   optimizer_teacher, optimizer_student,
                                   class_specific=class_specific, coefs=coefs, log=log, epoch=epoch,
                                   train_type=train_type)
    if train_type == 'retraining_globalnet':  # 2nd stage
        assert (optimizer_teacher is not None)
        model_teacher.train()
        model_student.eval()
        return _retraining_globalnet(model_teacher, model_student, labeled_loader, unlabeled_loader,
                                     optimizer_teacher, optimizer_student,
                                     class_specific=class_specific, coefs=coefs, log=log, epoch=epoch,
                                     train_type=train_type)


def test(model_teacher, model_student, dataloader, class_specific=False, log=print, epoch=None, train_type=None):

    log('\ttest')

    if train_type == 'pretraining_globalnet':
        model_teacher.eval()
        return _testing_globalnet(model=model_teacher, dataloader=dataloader, class_specific=class_specific,
                                  log=log, epoch=epoch)
    if train_type == 'training_protopnet':
        model_teacher.eval()
        model_student.eval()
        return _testing_twobranches(model_teacher, model_student, dataloader=dataloader, class_specific=class_specific,
                                    log=log, epoch=epoch)
    if train_type == 'retraining_globalnet':
        model_teacher.eval()
        model_student.eval()
        return _testing_twobranches(model_teacher, model_student, dataloader=dataloader, class_specific=class_specific,
                                    log=log, epoch=epoch)


def valid(model_teacher, model_student, dataloader, class_specific=False, log=print, epoch=None, train_type=None, valid_loss_min=None):

    log('\tvalid')

    if train_type == 'pretraining_globalnet':
        model_teacher.eval()
        return _validation_globalnet(model=model_teacher, dataloader=dataloader, class_specific=class_specific,
                                     log=log, epoch=epoch, valid_loss_min=valid_loss_min)
    if train_type == 'training_protopnet':
        model_teacher.eval()
        model_student.eval()
        return _validation_protopnet(model_teacher, model_student, dataloader=dataloader, class_specific=class_specific,
                                     log=log, epoch=epoch, valid_loss_min=valid_loss_min)
    if train_type == 'retraining_globalnet':
        model_teacher.eval()
        return _validation_globalnet(model=model_teacher, dataloader=dataloader, class_specific=class_specific,
                                     log=log, epoch=epoch, valid_loss_min=valid_loss_min)


def global_pretraining(model, log=print):

    # backbone
    for p in model.module._conv_stem_backbone.parameters():
        p.requires_grad = True
    for p in model.module._bn0_backbone.parameters():
        p.requires_grad = True
    for p in model.module._swish_backbone.parameters():
        p.requires_grad = True

    # teacher
    for name, p in model.module.features.named_parameters():
        p.requires_grad = True
    for p in model.module.global_classification_layers.parameters():
        p.requires_grad = True

    log('\tglobalnet pretraining')


def explain_backbone_only(model_teacher, model_student, log=print):
    # backbone
    for p in model_teacher.module._conv_stem_backbone.parameters():
        p.requires_grad = False
    for p in model_teacher.module._bn0_backbone.parameters():
        p.requires_grad = False
    for p in model_teacher.module._swish_backbone.parameters():
        p.requires_grad = False

    # teacher
    for p in model_teacher.module.features.parameters():
        p.requires_grad = True
    for p in model_teacher.module.global_classification_layers.parameters():
        p.requires_grad = True

    # student
    for p in model_student.module.features.parameters():
        p.requires_grad = True
    for p in model_student.module.add_on_layers.parameters():  # [512-->128-->128]
        p.requires_grad = True
    model_student.module.prototype_vectors.requires_grad = True  # [2000, 128, 1, 1]
    for p in model_student.module.last_layer_neg.parameters():  # [2000, 200]
        p.requires_grad = False
    for p in model_student.module.last_layer_pos.parameters():  # [2000, 200]
        p.requires_grad = False

    log('\tprotopnet backbone')


def explain_last_only(model_teacher, model_student, log=print):
    # backbone
    for p in model_teacher.module._conv_stem_backbone.parameters():
        p.requires_grad = False
    for p in model_teacher.module._bn0_backbone.parameters():
        p.requires_grad = False
    for p in model_teacher.module._swish_backbone.parameters():
        p.requires_grad = False

    # teacher
    for p in model_teacher.module.features.parameters():
        p.requires_grad = True
    for p in model_teacher.module.global_classification_layers.parameters():
        p.requires_grad = True

    # student
    for p in model_student.module.features.parameters():
        p.requires_grad = False
    for p in model_student.module.add_on_layers.parameters():  # [512-->128-->128]
        p.requires_grad = False
    model_student.module.prototype_vectors.requires_grad = False  # [2000, 128, 1, 1]
    for p in model_student.module.last_layer_neg.parameters():  # [2000, 200]
        p.requires_grad = True  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    for p in model_student.module.last_layer_pos.parameters():  # [2000, 200]
        p.requires_grad = True  # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    log('\tprotopnet last layer')


def explain_joint(model_teacher, model_student, log=print):
    # backbone
    for p in model_teacher.module._conv_stem_backbone.parameters():
        p.requires_grad = False
    for p in model_teacher.module._bn0_backbone.parameters():
        p.requires_grad = False
    for p in model_teacher.module._swish_backbone.parameters():
        p.requires_grad = False

    # teacher
    for p in model_teacher.module.features.parameters():
        p.requires_grad = True
    for p in model_teacher.module.global_classification_layers.parameters():
        p.requires_grad = True

    # student
    for p in model_student.module.features.parameters():
        p.requires_grad = True
    for p in model_student.module.add_on_layers.parameters():  # [512-->128-->128]
        p.requires_grad = True
    model_student.module.prototype_vectors.requires_grad = True  # [2000, 128, 1, 1]
    for p in model_student.module.last_layer_neg.parameters():  # [2000, 200]
        p.requires_grad = True
    for p in model_student.module.last_layer_pos.parameters():  # [2000, 200]
        p.requires_grad = True

    log('\tprotopnet joint')


def global_retraining(model_teacher, log=print):
    # backbone
    for p in model_teacher.module._conv_stem_backbone.parameters():
        p.requires_grad = False
    for p in model_teacher.module._bn0_backbone.parameters():
        p.requires_grad = False
    for p in model_teacher.module._swish_backbone.parameters():
        p.requires_grad = False

    # teacher
    for p in model_teacher.module.features.parameters():
        p.requires_grad = True
    for p in model_teacher.module.global_classification_layers.parameters():
        p.requires_grad = True

    log('\tretraining globalnet')