import torch
import torch.nn.functional as F
import numpy as np

from PIL import Image


imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])


@torch.no_grad()
def run_one_image(img, tgt, model, device):
    x = torch.tensor(img)
    # make it a batch-like
    x = torch.einsum("nhwc->nchw", x)

    tgt = torch.tensor(tgt)
    # make it a batch-like
    tgt = torch.einsum("nhwc->nchw", tgt)

    bool_masked_pos = torch.zeros(model.patch_embed.num_patches)
    bool_masked_pos[model.patch_embed.num_patches // 2 :] = 1
    bool_masked_pos = bool_masked_pos.unsqueeze(dim=0)
    valid = torch.ones_like(tgt)

    if model.seg_type == "instance":
        seg_type = torch.ones([valid.shape[0], 1])
    else:
        seg_type = torch.zeros([valid.shape[0], 1])

    feat_ensemble = 0 if len(x) > 1 else -1
    _, y, mask = model(
        x.float().to(device),
        tgt.float().to(device),
        bool_masked_pos.to(device),
        valid.float().to(device),
        seg_type.to(device),
        feat_ensemble,
    )
    y = model.unpatchify(y)
    y = torch.einsum("nchw->nhwc", y).detach().cpu()

    output = y[0, y.shape[1] // 2 :, :, :]
    output = torch.clip((output * imagenet_std + imagenet_mean) * 255, 0, 255)
    return output


def inference_image(model, device, image, prompt_images, prompt_targets):
    res, hres = 448, 448
    size = image.size
    image = np.array(image.resize((res, hres))) / 255.0

    image_batch, target_batch = [], []
    for img2, tgt2 in zip(prompt_images, prompt_targets):
        img2 = img2.resize((res, hres))
        img2 = np.array(img2) / 255.0

        tgt2 = tgt2.resize((res, hres), Image.NEAREST)
        tgt2 = np.array(tgt2) / 255.0

        tgt = tgt2  # tgt is not available
        tgt = np.concatenate((tgt2, tgt), axis=0)
        img = np.concatenate((img2, image), axis=0)

        assert img.shape == (2 * res, res, 3), f"{img.shape}"
        # normalize by ImageNet mean and std
        img = img - imagenet_mean
        img = img / imagenet_std

        assert tgt.shape == (2 * res, res, 3), f"{img.shape}"
        # normalize by ImageNet mean and std
        tgt = tgt - imagenet_mean
        tgt = tgt / imagenet_std

        image_batch.append(img)
        target_batch.append(tgt)

    img = np.stack(image_batch, axis=0)
    tgt = np.stack(target_batch, axis=0)
    """### Run SegGPT on the image"""
    # make random mask reproducible (comment out to make it change)
    torch.manual_seed(2)
    output = run_one_image(img, tgt, model, device)
    output = (
        F.interpolate(
            output[None, ...].permute(0, 3, 1, 2),
            size=[size[1], size[0]],
            mode="nearest",
        )
        .permute(0, 2, 3, 1)[0]
        .numpy()
    )

    return output
