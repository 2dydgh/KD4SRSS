def calculate_margin(feature_map, eps=1e-6):
    """
    Calculate the margin based on the negative values in the feature map.

    Args:
        feature_map (torch.Tensor): Input feature map from teacher network.
        eps (float): Small value to prevent division by zero.

    Returns:
        torch.Tensor: Calculated margin.
    """
    mask = (feature_map < 0.0).float()
    masked_feature = feature_map * mask
    margin = masked_feature.sum(dim=(0, 2, 3), keepdim=True) / (mask.sum(dim=(0, 2, 3), keepdim=True) + eps)
    return margin
