import torch
import numpy as np
from torchvision import datasets, models, transforms


def sample_z(batch_size, z_dim=128, dist="uniform"):
    if dist == "uniform":
        return torch.from_numpy(np.random.uniform(-1., 1., (batch_size, z_dim))).float()
    else:
        return torch.from_numpy(np.random.normal(0., 1., (batch_size, z_dim))).float()


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    if len(size) == 4:
        feat_std = feat_var.sqrt().view(N, C, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    elif len(size) == 5:
        feat_std = feat_var.sqrt().view(N, C, 1, 1, 1)
        feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1, 1)
    else:
        assert 1 == 0
    return feat_mean, feat_std


def adaptive_instance_normalization(features, style_feat):
    partition = style_feat.size()[1] // 2
    scale, bias = style_feat[:, :partition], style_feat[:, partition:]
    mean, variance = calc_mean_std(features)  # Only consider spatial dimension
    sigma = torch.rsqrt(variance + 1e-8)
    normalized = (features - mean) * sigma
    scale_broadcast = scale.view(mean.size())
    bias_broadcast = bias.view(mean.size())
    normalized = scale_broadcast * normalized
    normalized += bias_broadcast
    return normalized


def rot_matrix_x(theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3,3)).astype(np.float32)
    mat[0, 0] = 1.
    mat[1, 1] = np.cos(theta)
    mat[1, 2] = -np.sin(theta)
    mat[2, 1] = np.sin(theta)
    mat[2, 2] = np.cos(theta)
    return mat


def rot_matrix_y(theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3,3)).astype(np.float32)
    mat[0, 0] = np.cos(theta)
    mat[0, 2] = np.sin(theta)
    mat[1, 1] = 1.
    mat[2, 0] = -np.sin(theta)
    mat[2, 2] = np.cos(theta)
    return mat


def rot_matrix_z(theta):
    """
    theta: measured in radians
    """
    mat = np.zeros((3,3)).astype(np.float32)
    mat[0, 0] = np.cos(theta)
    mat[0, 1] = -np.sin(theta)
    mat[1, 0] = np.sin(theta)
    mat[1, 1] = np.cos(theta)
    mat[2, 2] = 1.
    return mat


def pad_rotmat(theta):
    """theta = (3x3) rotation matrix"""
    return np.hstack((theta, np.zeros((3, 1))))


def get_theta(angles):
    bs = len(angles)
    theta = np.zeros((bs, 3, 4))

    angles_yaw = angles[:, 0]
    angles_pitch = angles[:, 1]
    angles_roll = angles[:, 2]
    for i in range(bs):
        theta[i] = pad_rotmat(
            np.dot(np.dot(rot_matrix_z(angles_roll[i]), rot_matrix_y(angles_pitch[i])), rot_matrix_x(angles_yaw[i]))
        )

    return torch.from_numpy(theta).float()


def sample_angles(bs,
                  min_angle_yaw,
                  max_angle_yaw,
                  min_angle_pitch,
                  max_angle_pitch,
                  min_angle_roll,
                  max_angle_roll):
    """Sample random yaw, pitch, and roll angles"""
    angles = []
    for i in range(bs):
        rnd_angles = [
            np.random.uniform(min_angle_yaw, max_angle_yaw),
            np.random.uniform(min_angle_pitch, max_angle_pitch),
            np.random.uniform(min_angle_roll, max_angle_roll),
        ]
        angles.append(rnd_angles)
    return np.asarray(angles)


def to_radians(deg):
    return deg * (np.pi / 180)


def angles_to_dict(angles):
    angles = {
        'min_angle_yaw': to_radians(angles[0]),
        'max_angle_yaw': to_radians(angles[1]),
        'min_angle_pitch': to_radians(angles[2]),
        'max_angle_pitch': to_radians(angles[3]),
        'min_angle_roll': to_radians(angles[4]),
        'max_angle_roll': to_radians(angles[5])
    }
    return angles


rot2idx = {
    'yaw': 0,
    'pitch': 1,
    'roll': 2
}


def binary_loss(prediction, target):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction) * target
        if prediction.is_cuda:
            target = target.cuda()
    loss = torch.nn.BCEWithLogitsLoss()
    if prediction.is_cuda:
        loss = loss.cuda()
    return loss(prediction, target)


def mse_loss(prediction, target):
    if not hasattr(target, '__len__'):
        target = torch.ones_like(prediction) * target
        if prediction.is_cuda:
            target = target.cuda()
    loss = torch.nn.MSELoss()
    if prediction.is_cuda:
        loss = loss.cuda()
    return loss(prediction, target)


def get_data_loader(data_dir, batch_size, num_workers=4):
    dataset = datasets.ImageFolder(data_dir, transforms.Compose([transforms.RandomResizedCrop(128), transforms.ToTensor(),transforms.Normalize([0., 0., 0.], [1., 1., 1.])]))
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    return data_loader, len(dataset)