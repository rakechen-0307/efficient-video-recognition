from model import EVLTransformer
import torch
import av
from PIL import Image
import numpy as np
import torch.distributed as dist
from torchvision import transforms
from transform import create_random_augment, random_resized_crop

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
num_workers = 12
batch_split = 1

class VideoDataset(torch.utils.data.Dataset):
    def __init__(
        self, data, num_spatial_views, num_temporal_views,
        random_sample, num_frames, sampling_rate, spatial_size,
        mean, std, auto_augment = None, interpolation = 'bicubic', 
        mirror = False
    ):
        self.data = data
        self.interpolation = interpolation
        self.spatial_size = spatial_size

        self.mean, self.std = mean, std
        self.num_frames, self.sampling_rate = num_frames, sampling_rate

        if random_sample:
            assert num_spatial_views == 1 and num_temporal_views == 1
            self.random_sample = True
            self.mirror = mirror
            self.auto_augment = auto_augment
        else:
            assert auto_augment is None and not mirror
            self.random_sample = False
            self.num_temporal_views = num_temporal_views
            self.num_spatial_views = num_spatial_views

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        path = self.data[idx][0]
        label = self.data[idx][1]

        container = av.open(path)
        frames = {}
        for frame in container.decode(video=0):
            frames[frame.pts] = frame
        container.close()
        frames = [frames[k] for k in sorted(frames.keys())]
        frame_idx = []
        for i in range(self.num_frames):
            frame_idx.append(i * self.sampling_rate if i * self.sampling_rate < len(frames) else frame_idx[-1])

        cropped_frames = []
        for x in frame_idx:
            img = frames[x].to_image()  # PIL image
            width, height = img.size   # Get dimensions

            new_size = min(width, height)
            left = (width - new_size) // 2
            top = (height - new_size) // 2
            right = left + new_size
            bottom = top + new_size
            img = img.crop((left, top, right, bottom))  # Crop the center of the image

            cropped_frame = av.video.frame.VideoFrame.from_image(img).reformat(width=self.spatial_size, height=self.spatial_size).to_rgb().to_ndarray()
            cropped_frames.append(cropped_frame)

        frames = cropped_frames
        frames = torch.as_tensor(np.stack(frames)).float() / 255.
        frames = (frames - self.mean) / self.std
        frames = frames.permute(3, 0, 1, 2) # C, T, H, W

        return frames, int(label)

    def _generate_temporal_crops(self, frames):
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

    def _generate_spatial_crops(self, frames):
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
         
    def _random_sample_frame_idx(self, len):
        frame_indices = []

        if self.sampling_rate < 0: # tsn sample
            seg_size = (len - 1) / self.num_frames
            for i in range(self.num_frames):
                start, end = round(seg_size * i), round(seg_size * (i + 1))
                frame_indices.append(np.random.randint(start, end + 1))
        elif self.sampling_rate * (self.num_frames - 1) + 1 >= len:
            for i in range(self.num_frames):
                frame_indices.append(i * self.sampling_rate if i * self.sampling_rate < len else frame_indices[-1])
        else:
            start = np.random.randint(len - self.sampling_rate * (self.num_frames - 1))
            frame_indices = list(range(start, start + self.sampling_rate * self.num_frames, self.sampling_rate))

        return frame_indices  

def Loader(data):
    rank, world_size = (0, 1) if not dist.is_initialized() else (dist.get_rank(), dist.get_world_size())
    # sampler for distribued eval
    sampler = list(range(rank, len(data), world_size))

    loader = torch.utils.data.DataLoader(
        data, sampler=sampler, batch_size=1,
        num_workers=num_workers, pin_memory=False,
    )
    return loader

model = EVLTransformer(
    num_frames=num_frames,
    backbone_name="ViT-L/14-lnpre",
    backbone_type="clip",
    backbone_path="../ViT-L-14.pt",
    backbone_mode="freeze_fp16",
    decoder_num_layers=decoder_num_layers,
    decoder_qkv_dim=decoder_qkv_dim,
    decoder_num_heads=decoder_num_heads,
    num_classes=400
)
model.cuda()
model.eval()

data = [("./001-1.mp4", 0)]
dataset = VideoDataset(data=data, num_spatial_views=1,
                       num_temporal_views=1, random_sample=True,
                       num_frames=num_frames, sampling_rate=sampling_rate, spatial_size=spatial_size,
                       mean=mean, std=std)
loader = Loader(dataset)
criterion = torch.nn.CrossEntropyLoss()
for data, labels in loader:
    data, labels = data.cuda(), labels.cuda()

    with torch.no_grad():
        logits = model(data)
        # print(logits)