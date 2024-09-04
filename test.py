from model import EVLTransformer
import torch
import av
import numpy as np

mean = torch.Tensor([0.48145466, 0.4578275, 0.40821073])
std = torch.Tensor([0.26862954, 0.26130258, 0.27577711])
spatial_size = 224
num_frames = 32
sampling_rate = 8
num_temporal_views = 1
num_spatial_views = 3
decoder_num_layers = 4
decoder_qkv_dim = 1024
decoder_num_heads = 16

def generate_temporal_crops(frames):
    seg_len = (num_frames - 1) * sampling_rate + 1
    if frames.size(1) < seg_len:
        frames = torch.cat([frames, frames[:, -1:].repeat(1, seg_len - frames.size(1), 1, 1)], dim=1)
    slide_len = frames.size(1) - seg_len

    crops = []
    for i in range(num_temporal_views):
        if num_temporal_views == 1:
            st = slide_len // 2
        else:
            st = round(slide_len / (num_temporal_views - 1) * i)

        crops.append(frames[:, st: st + num_frames * sampling_rate: sampling_rate])
    
    return crops

def generate_spatial_crops(frames):
    if num_spatial_views == 1:
        assert min(frames.size(-2), frames.size(-1)) >= spatial_size
        h_st = (frames.size(-2) - spatial_size) // 2
        w_st = (frames.size(-1) - spatial_size) // 2
        h_ed, w_ed = h_st + spatial_size, w_st + spatial_size
        return [frames[:, :, h_st: h_ed, w_st: w_ed]]

    elif num_spatial_views == 3:
        assert min(frames.size(-2), frames.size(-1)) == spatial_size
        crops = []
        margin = max(frames.size(-2), frames.size(-1)) - spatial_size
        for st in (0, margin // 2, margin):
            ed = st + spatial_size
            if frames.size(-2) > frames.size(-1):
                crops.append(frames[:, :, st: ed, :])
            else:
                crops.append(frames[:, :, :, st: ed])
        return crops

def DataPrepare(path):
    container = av.open(path)
    frames = {}
    for frame in container.decode(video=0):
        frames[frame.pts] = frame
    container.close()
    frames = [frames[k] for k in sorted(frames.keys())]

    frames = [x.to_rgb().to_ndarray() for x in frames]
    frames = torch.as_tensor(np.stack(frames), dtype=torch.float)
    frames = frames / 255.

    frames = (frames - mean) / std
    frames = frames.permute(3, 0, 1, 2) # C, T, H, W
    
    if frames.size(-2) < frames.size(-1):
        new_width = frames.size(-1) * spatial_size // frames.size(-2)
        new_height = spatial_size
    else:
        new_height = frames.size(-2) * spatial_size // frames.size(-1)
        new_width = spatial_size
    frames = torch.nn.functional.interpolate(
        frames, size=(new_height, new_width),
        mode='bilinear', align_corners=False,
    )

    frames = generate_spatial_crops(frames)
    frames = sum([generate_temporal_crops(x) for x in frames], [])
    if len(frames) > 1:
        frames = torch.stack(frames)

    return frames

video_path = "./001-1.mp4"
frames = DataPrepare(video_path)

model = EVLTransformer(
    num_frames=num_frames,
    backbone_name="ViT-L/14-lnpre",
    backbone_path="./ViT-L-14.pt",
    backbone_mode="freeze_fp16",
    decoder_num_layers=decoder_num_layers,
    decoder_qkv_dim=decoder_qkv_dim,
    decoder_num_heads=decoder_num_heads,
    num_classes=400
)
model.cuda()
logits = model(frames)
print(logits)