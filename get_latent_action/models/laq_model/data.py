from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader as PytorchDataLoader
from torchvision import transforms as T
import os
import random
import cv2
import glob
import av
import tensorflow_datasets as tfds
import imageio
import sys
import numpy as np
import glob 
from tqdm import tqdm

def get_dataset_path(parent_dir, dataset_name):
    if dataset_name == 'robo_net' or dataset_name == 'cmu_playing_with_food':
        version = '1.0.0'
    elif dataset_name == 'language_table':
        version = '0.0.1'
    elif dataset_name[:-1] == 'uiuc_d3field' or dataset_name[:-1] == 'stanford_robocook_converted_externally_to_rlds':
        dataset_name = dataset_name[:-1]
        version = '0.1.0'
    else:
        version = '0.1.0'
    return os.path.join(parent_dir, dataset_name, version)

def get_sharded_dataset(builder_dir, num_shards, shard_index):
    """加载分片数据集"""
    builder = tfds.builder_from_directory(builder_dir)
    ds = {}
    for split in builder.info.splits:  # 遍历所有数据分割
        full_ds = builder.as_dataset(split=split)
        # 转换为可迭代数据集并分片
        sharded_ds = full_ds.shard(num_shards=num_shards, index=shard_index)
        ds[split] = sharded_ds
    return ds

def get_video_info(video_path):
    """获取视频总帧数和时长"""
    with av.open(video_path) as container:
        video_stream = next(s for s in container.streams if s.type == 'video')
        return {
            "total_frames": video_stream.frames,
            "duration": float(video_stream.duration * video_stream.time_base),
            "fps": float(video_stream.average_rate)
        }

def get_kth_pil_frame(video_path: str, k: int) -> Image.Image:
    """高效获取第k帧的PIL Image（支持精确跳转）"""
    with av.open(video_path) as container:
        video_stream = next(s for s in container.streams if s.type == 'video')
        
        # 精确跳转到指定帧
        container.seek(offset=k, stream=video_stream)
        
        for frame in container.decode(video_stream):
            # 直接转换为RGB格式的PIL Image
            return frame.to_image()  # 原生支持PIL转换
    return None

def get_all_pil_frame(video_path: str) -> Image.Image:
    all_frames = []
    with av.open(video_path) as container:
        video_stream = next(s for s in container.streams if s.type == 'video')
        
        container.seek(offset=0, stream=video_stream)
        
        for frame in container.decode(video_stream):
            # 直接转换为RGB格式的PIL Image
            all_frames.append(frame.to_image())  # 原生支持PIL转换
    return all_frames

def get_video_frames_as_pil(video_path):
    """
    读取MP4文件获取所有PIL Image帧列表
    :param video_path: 视频文件路径
    :return: PIL.Image列表
    """
    cap = cv2.VideoCapture(video_path, cv2.CAP_FFMPEG)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件：{video_path}")

    pil_frames = []
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
                
            # 关键转换步骤
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(rgb_frame)
            pil_frames.append(pil_image)
            
    finally:
        cap.release()
    
    return pil_frames

def read_av1_frames(video_path):
    """
    读取AV1编码的MP4文件并返回PIL图像列表
    """
    container = av.open(video_path)
    pil_frames = []

    # 配置解码参数
    codec_context = av.CodecContext.create('av1', 'r')

    try:
        for packet in container.demux():
            if packet.stream.type == 'video':
                # 解码数据包
                frames = codec_context.decode(packet)
                
                for frame in frames:
                    # 转换为RGB24格式
                    rgb_frame = frame.reformat(format='rgb24')
                    
                    # 转换为PIL Image
                    pil_image = Image.frombytes(
                        'RGB',
                        (rgb_frame.width, rgb_frame.height),
                        rgb_frame.planes[0].to_bytes()
                    )
                    pil_frames.append(pil_image)
    finally:
        container.close()
    
    return pil_frames

def exists(val):
    return val is not None

def identity(t, *args, **kwargs):
    return t

def pair(val):
    return val if isinstance(val, tuple) else (val, val)

'''
This is the dataset class for Sthv2 dataset.
The dataset is a list of folders, each folder contains a sequence of frames.
You have to change the dataset class to fit your dataset for custom training.
'''
# 需要满足是一个三级目录，folder下是frame的文件夹list，frame进去是对应的img的文件list
class ValidImageVideoDataset(Dataset):
    def __init__(
        self,
        image_size
    ):
        super().__init__()
        print("使用img size：", image_size)
        self.seed = torch.randint(0, 2**32-1, ()).item()  # 直接生成32位安全种子
        self.image_size = image_size
      
        self.dataset_names = [
            "libero_imgs/libero_10_no_noops",
            "oxe_imgs/austin_sailor_dataset_converted_externally_to_rlds", # 选做validation好了
            "mixkit_pixabay_pexels_videovo_merge_extract_imgs" # human 组
        ]
        hl_path = os.environ.get("LAQ_DATASET_DIR", "")
        
        self.dataset_files = []
        for dataset in tqdm(self.dataset_names):
            if dataset in ["libero_imgs/libero_10_no_noops", "oxe_imgs/austin_sailor_dataset_converted_externally_to_rlds"]:
                self.dataset_files.append(glob.glob(f"{hl_path}/{dataset}/*/*", recursive=True))
            elif dataset == "mixkit_pixabay_pexels_videovo_merge_extract_imgs":
                self.dataset_files.append(glob.glob(f"{hl_path}/{dataset}/*/*/*", recursive=True))
        
        self.dataset_weights = [
            0.4,
            0.3,
            0.3
        ]

        self.dataset_offsets = [
            20, # libero_10_no_noops
            60, # austin_sailor_dataset_converted_externally_to_rlds
            1, # "mixkit_pixabay_pexels_videovo_merge_extract_imgs" # human 组 已经完成了抽帧
        ]
        assert len(self.dataset_names) == len(self.dataset_weights) == len(self.dataset_offsets)

        # print all datasets info
        print("=============================================")
        print("validation dataset content:")
        for idx in range(len(self.dataset_names)):
            name = self.dataset_names[idx]
            offset = self.dataset_offsets[idx]
            num_episodes = len(self.dataset_files[idx])
            print(f"{name} has {num_episodes} episodes using offset as {offset}")
        print("=============================================")

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        return 64_000_000 # 让这个不起作用
    
    def __getitem__(self, index):
        try :
            # step1. 根据预先分配的权重随机选择数据集
            safe_seed = (self.seed + index) % (2**32)
            local_random = random.Random(safe_seed)
            local_np_random = np.random.RandomState(safe_seed)
            dataset_idx = local_np_random.choice(
                len(self.dataset_weights),
                p=self.dataset_weights
            )
            current_dataset = self.dataset_files[dataset_idx]

            offset = self.dataset_offsets[dataset_idx]

            # step2. 视频随机采样
            frame_folder = local_random.choice(current_dataset)

            # step3. 随机采样前后帧            
            img_list = os.listdir(frame_folder)
            img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))
            total_frames = len(img_list)

            ## pick random first frame 
            first_frame_idx = random.randint(0, total_frames-1)
            # pick random next frame
            second_frame_idx = random.randint(first_frame_idx + offset, first_frame_idx + int(offset * 1.5))

            first_frame_idx = min(first_frame_idx, total_frames-1)
            second_frame_idx = min(second_frame_idx, total_frames-1)

            first_path = os.path.join(frame_folder, img_list[first_frame_idx])
            second_path = os.path.join(frame_folder, img_list[second_frame_idx])
                    
            img = Image.open(first_path)
            next_img = Image.open(second_path)
            
            transform_img = self.transform(img).unsqueeze(1)
            next_transform_img = self.transform(next_img).unsqueeze(1)
            
            cat_img = torch.cat([transform_img, next_transform_img], dim=1)
            return cat_img
        except :
            print("error", index)
            return self.__getitem__(index + 1) # 重新随机生成


'''
This is the dataset class for Sthv2 dataset.
The dataset is a list of folders, each folder contains a sequence of frames.
You have to change the dataset class to fit your dataset for custom training.
'''
# 需要满足是一个三级目录，folder下是frame的文件夹list，frame进去是对应的img的文件list
class ImageVideoDataset(Dataset):
    def __init__(
        self,
        image_size
    ):
        super().__init__()
        print("使用img size：", image_size)
        self.seed = torch.randint(0, 2**32-1, ()).item()  # 直接生成32位安全种子
        print("使用seed: ", self.seed)
        self.image_size = image_size
      
        self.dataset_names = [
            "oxe_imgs/fractal20220817_data",
            "oxe_imgs/kuka",
            "oxe_imgs/bridge",
            "oxe_imgs/taco_play",
            "oxe_imgs/roboturk",
            "oxe_imgs/toto",
            "oxe_imgs/stanford_hydra_dataset_converted_externally_to_rlds",
            "oxe_imgs/furniture_bench_dataset_converted_externally_to_rlds",
            "oxe_imgs/utaustin_mutex",
            "oxe_imgs/bc_z",

            "egovid_extracted_imgs", # human 组
            "artgrid_extract_imgs", # human 组

            "agibot_beta_extracted_imgs",
            "panda_imgs_extracted" # general video 组
        ]
        hl_path = os.environ.get("LAQ_DATASET_DIR", "")
        zw_path = os.environ.get("LAQ_DATASET_DIR_ZW", "")

        # 当前的panda图像还是存在zw，所以分开来看；而且有的是三级，有的是两级目录
        self.dataset_files = []
        for dataset in tqdm(self.dataset_names):
            if dataset.startswith("oxe") or dataset == "agibot_beta_extracted_imgs": 
                # robot data; hl, 两级目录
                self.dataset_files.append(glob.glob(f"{hl_path}/{dataset}/*/*", recursive=True))
            elif dataset in ["egovid_extracted_imgs", "artgrid_extract_imgs"]:
                # human data; hl, 三级目录
                self.dataset_files.append(glob.glob(f"{hl_path}/{dataset}/*/*/*", recursive=True))
            elif dataset == 'panda_imgs_extracted':
                # general data; zw, 两级目录
                # self.dataset_files.append([])
                self.dataset_files.append(glob.glob(f"{zw_path}/{dataset}/*/*", recursive=True))
        
        # 满足 （OXE+libero）: HUMAN : AgiBot-Beta : General video = 1: 1: 2: 1
        self.dataset_weights = [
            0.20 * 0.2, # fractal20220817_data
            0.40 * 0.2, # kuka
            0.18 * 0.2, # bridge
            0.04 * 0.2, # taco_play
            0.02 * 0.2, # roboturk
            0.02 * 0.2, # toto
            0.04 * 0.2, # stanford_hydra_dataset_converted_externally_to_rlds
            0.02 * 0.2, # furniture_bench_dataset_converted_externally_to_rlds
            0.02 * 0.2, # utaustin_mutex
            0.06 * 0.2, # bc_z

            0.90 * 0.2, # "egovid_extracted_imgs" # human 组
            0.10 * 0.2, # "artgrid_extract_imgs" # human 组

            2 * 0.2, # agibot_beta_extracted_imgs
            1 * 0.2, # panda_imgs_extracted
        ]

        self.dataset_offsets = [
            10, # fractal20220817_data
            2, # kuka
            5 , # bridge_v2
            9, # taco_play
            3, # roboturk
            30, # toto
            20, # stanford_hydra_dataset_converted_externally_to_rlds
            20, # furniture_bench_dataset_converted_externally_to_rlds
            20, # utaustin_mutex
            30, # bc_z
            
            1, # "egovid_extracted_imgs" # human 组 已经完成了抽帧
            1, # "artgrid_extract_imgs" # human 组  已经完成了抽帧
            
            1, # agibot_beta_extracted_imgs，已经完成了抽帧
            1, # panda 已经完成了抽帧
        ]
        assert len(self.dataset_names) == len(self.dataset_weights) == len(self.dataset_offsets)

        # print all datasets info
        print("=============================================")
        for idx in range(len(self.dataset_names)):
            name = self.dataset_names[idx]
            offset = self.dataset_offsets[idx]
            num_episodes = len(self.dataset_files[idx])
            print(f"{name} has {num_episodes} episodes using offset as {offset}")
        print("=============================================")

        self.transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize(image_size),
            T.ToTensor(),
        ])

    def __len__(self):
        # return len(self.folder_list) ## length of folder list is not exact number of frames; TODO: change this to actual number of frames
        return 64_000_000 # 让这个不起作用

    def __getitem__(self, index):
        try :
            # step1. 根据预先分配的权重随机选择数据集
            safe_seed = (self.seed + index) % (2**32)
            local_random = random.Random(safe_seed)
            local_np_random = np.random.RandomState(safe_seed)
            dataset_idx = local_np_random.choice(
                len(self.dataset_weights),
                p=self.dataset_weights
            )
            current_dataset = self.dataset_files[dataset_idx]

            offset = self.dataset_offsets[dataset_idx]

            # step2. 视频随机采样
            frame_folder = local_random.choice(current_dataset)

            # step3. 随机采样前后帧            
            img_list = os.listdir(frame_folder)
            img_list = sorted(img_list, key=lambda x: int(x.split('.')[0]))
            total_frames = len(img_list)

            ## pick random first frame 
            first_frame_idx = random.randint(0, total_frames-1)
            # pick random next frame
            second_frame_idx = random.randint(first_frame_idx + offset, first_frame_idx + int(offset * 1.5))

            first_frame_idx = min(first_frame_idx, total_frames-1)
            second_frame_idx = min(second_frame_idx, total_frames-1)

            first_path = os.path.join(frame_folder, img_list[first_frame_idx])
            second_path = os.path.join(frame_folder, img_list[second_frame_idx])
                    
            img = Image.open(first_path)
            next_img = Image.open(second_path)
            
            transform_img = self.transform(img).unsqueeze(1)
            next_transform_img = self.transform(next_img).unsqueeze(1)
            
            cat_img = torch.cat([transform_img, next_transform_img], dim=1)
            return cat_img
        except :
            print("error", index)
            return self.__getitem__(index + 1) # 重新随机生成

# '''
# This is the dataset class for Sthv2 dataset.
# The dataset is a list of folders, each folder contains a sequence of frames.
# You have to change the dataset class to fit your dataset for custom training.
# '''
# DISPLAY_KEY = {
#     'taco_play': 'rgb_static',
#     'roboturk': 'front_rgb',
#     'viola': 'agentview_rgb',
#     'language_table': 'rgb',
#     'stanford_robocook_converted_externally_to_rlds1': 'image_1',
#     'stanford_robocook_converted_externally_to_rlds2': 'image_2',
#     'stanford_robocook_converted_externally_to_rlds3': 'image_3',
#     'stanford_robocook_converted_externally_to_rlds4': 'image_4',
#     'uiuc_d3field1': 'image_1',
#     'uiuc_d3field2': 'image_2',
#     'uiuc_d3field3': 'image_3',
#     'uiuc_d3field4': 'image_4',
# }

# class OXEImageVideoDataset(Dataset):
#     def __init__(
#         self,
#         folder,
#         image_size,
#         offset=5,
#         offset_end=10,
#     ):
#         super().__init__()
        
#         self.folder = folder
#         self.folder_list = dataset_files = glob.glob(f"{folder}/*/*", recursive=True)
#         self.image_size = image_size

#         dataset_name = "fractal20220817_data"
#         self.ds = tfds.builder_from_directory(builder_dir=get_dataset_path(folder, dataset_name)).as_dataset()
#         self.display_key = DISPLAY_KEY.get(dataset_name, 'image')
#         self.total_frames = len(self.ds['train'])

#         self.offset = offset
#         self.offset_end = offset_end
#         assert self.offset <= self.offset_end, "self.offset > self.offset_end!"

#         self.transform = T.Compose([
#             T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
#             T.Resize(image_size),
#             T.ToTensor(),
#         ])
#         breakpoint()

#     def __len__(self):
#         return self.total_frames ## length of folder list is not exact number of frames; TODO: change this to actual number of frames
    
#     def __getitem__(self, index):
#         # try :
#         # 获取当前视频数据
#         # breakpoint()

#         episode = next(iter(self.ds['train'].skip(index)))
#         all_steps = list(episode['steps'].as_numpy_iterator())
#         total_frames = len(all_steps)

#         first_frame_idx = random.randint(0, total_frames-1)
#         second_frame_idx = random.randint(first_frame_idx+self.offset, first_frame_idx+self.offset_end)

#         first_frame_idx = min(first_frame_idx, total_frames-1)
#         second_frame_idx = min(second_frame_idx, total_frames-1)
        
#         img = Image.fromarray(all_steps[first_frame_idx]['observation'][self.display_key], mode='RGB')
#         next_img = Image.fromarray(all_steps[second_frame_idx]['observation'][self.display_key], mode='RGB')


#         transform_img = self.transform(img).unsqueeze(1)
#         next_transform_img = self.transform(next_img).unsqueeze(1)
        
#         cat_img = torch.cat([transform_img, next_transform_img], dim=1)
#         return cat_img
#         # except :
#         #     print("error", index)
#         #     if index < self.__len__() - 1:
#         #         return self.__getitem__(index + 1)
#         #     else:
#         #         return self.__getitem__(random.randint(0, self.__len__() - 1))
