import torch


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
    # print(features.size())
    # print(style_feat.size())
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

# def adaptive_instance_normalization(content_feat, style_feat):
#     assert (content_feat.size()[:2] == style_feat.size()[:2])
#     size = content_feat.size()
#     style_mean, style_std = calc_mean_std(style_feat)
#     content_mean, content_std = calc_mean_std(content_feat)
#
#     normalized_feat = (content_feat - content_mean.expand(size)) / content_std.expand(size)
#     return normalized_feat * style_std.expand(size) + style_mean.expand(size)
