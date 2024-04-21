import os
import torch


def save_model_w_condition(model_teacher, model_student, model_dir, model_name, auc, target_auc, log=print):
    '''
    model: this is not the multigpu model
    '''
    if auc > target_auc:
        log('\tabove {0:.2f}%'.format(target_auc * 100))
        if model_student is not None:
            torch.save(model_student.state_dict(), os.path.join(model_dir, (model_name + '_S_' + '{0:.4f}.pth').format(auc)))
        torch.save(model_teacher.state_dict(), os.path.join(model_dir, (model_name + '_T_' + '{0:.4f}.pth').format(auc)))