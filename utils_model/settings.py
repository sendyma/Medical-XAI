base_architecture = 'efficientnet_b0'
img_size = 1536
num_classes = 2
prototype_shape = (num_classes * 200, 128, 1, 1)

prototype_activation_function = 'log'
add_on_layers_type = 'regular'

experiment_run = str(img_size)

data_path = '/mnt/c/data/'
train_dir = data_path + 'train/'
train_push_dir = data_path + 'train/'
test_dir = data_path + 'test/'
train_batch_size = 16        # 8
test_batch_size = 36         # 32
train_push_batch_size = 36   # 32

joint_optimizer_lrs = {'features': 1e-4,
                       'add_on_layers': 3e-3,
                       'prototype_vectors': 3e-3}

joint_lr_step_size = 10  # 5

warm_optimizer_lrs = {'add_on_layers': 3e-3,
                      'prototype_vectors': 3e-3}

last_layer_optimizer_lr = 1e-4

coefs = {
    'crs_ent': 1,
    # 'clst': 0.8,
    'clst': 0.1,
    'sep': -0.08,
    'dis': 0.1,
    'l1': 1e-4,
}

num_train_epochs = 50
num_warm_epochs = 0  # 5

push_start = 40   # 10
push_epochs = [i for i in range(num_train_epochs) if i % 5 == 0]  # 10
