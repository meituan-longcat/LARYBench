import os
import argparse
import numpy as np
import pandas as pd
import cv2
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from skimage.metrics import structural_similarity
import hashlib
# 假设 sampler.py 包含这两个类
from get_latent_action.sampler import UniformSampler, MGSampler


# ---------------------------------------------------------
# 1. 第一阶段：快速检测与特征提取 (只计算，不保存图)
# ---------------------------------------------------------
def extract_ssim_stage1(video_path, cache_path, num_frames_limit):
    """
    1. 快速获取帧数，若太短则存入特殊标记 [-1.0]
    2. 流式计算 SSIM 并降采样，节省内存
    """
    # 断点续传
    if os.path.exists(cache_path):
        return True

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Warning: Could not open video {video_path}")
        return False

    # [快速检测] 获取总帧数
    # 注意：某些特殊编码视频可能获取不准，但对大多数情况有效且极快
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 如果总帧数过少，存入特殊标记
    if total_frames > 0 and total_frames <= num_frames_limit:
        np.save(cache_path, np.array([-1.0]))
        cap.release()
        return True

    # [流式计算] 计算 SSIM
    diffs = []
    prev_gray = None
    i = 0 
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                break

            i += 1

            # 关键优化：转灰度并降采样到 224x224，加速 SSIM 计算
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_small = cv2.resize(gray, (224, 224))
            
            if prev_gray is not None:
                # 计算结构相似性
                score = structural_similarity(prev_gray, gray_small)
                diffs.append(float(1.0 - score))
            prev_gray = gray_small
        if i <= num_frames_limit:
            np.save(cache_path, np.array([-1.0]))
            return True
        np.save(cache_path, np.array(diffs))
        return True
    except Exception as e:
        print(f"Error processing SSIM for {video_path}: {e}")
        # 出错时删除可能损坏的文件，确保下次重跑
        if os.path.exists(cache_path): os.remove(cache_path)
        return False
    finally:
        cap.release()

def get_safe_cache_path(video_path, cache_dir):
    """
    根据视频的绝对路径生成唯一的 MD5 文件名
    """
    # 获取绝对路径，防止相对路径造成的歧义
    abs_path = os.path.abspath(video_path)
    # 计算 MD5 哈希
    path_hash = hashlib.md5(abs_path.encode('utf-8')).hexdigest()
    # 结合原始文件名（可选，方便肉眼识别）和哈希值
    v_name = os.path.basename(video_path).split('.')[0]
    # 限制 v_name 长度并去除特殊字符，防止文件名过长
    safe_v_name = "".join([c for c in v_name if c.isalnum()])[:10]
    
    return os.path.join(cache_dir, f"{safe_v_name}_{path_hash}.npy")

def stage1_worker(row, cache_dir, num_frames_limit):
    v_path = row['video_path']
    # 生成安全的文件名作为缓存ID
    cache_path = get_safe_cache_path(v_path, cache_dir)
    return extract_ssim_stage1(v_path, cache_path, num_frames_limit)

# ---------------------------------------------------------
# 2. 可视化函数 (跳帧读取)
# ---------------------------------------------------------
def save_visualization_jump(v_path, indices, action, vis_dir):
    """
    根据给定的索引，使用 seek 的方式跳跃读取特定帧并保存拼接图。
    比读取整个视频要快得多。
    """
    cap = cv2.VideoCapture(v_path)
    if not cap.isOpened(): return

    # 为了提高 seek 效率，先读取唯一的、排序后的帧
    unique_indices = sorted(list(set(indices)))
    frames_map = {}
    
    for idx in unique_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if ret:
            # 在帧上绘制索引号
            cv2.putText(frame, f"id:{idx}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            frames_map[idx] = frame
    cap.release()

    # 按照原始 indices 的顺序重新组装帧
    final_frames = []
    for idx in indices:
        if idx in frames_map:
            final_frames.append(frames_map[idx])
            
    if final_frames:
        # 水平拼接
        combined = np.hstack(final_frames)
        v_name = os.path.basename(v_path).split('.')[0]
        safe_action = str(action).replace(' ', '_').replace('/', '-')
        out_path = os.path.join(vis_dir, f"{safe_action}_{v_name}.jpg")
        cv2.imwrite(out_path, combined)

# ---------------------------------------------------------
# 3. 主程序
# ---------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    _LAB_DIR = os.environ.get("LATENT_ACTION_BENCH_DIR")
    parser.add_argument("--input_file", type=str, default=os.path.join(_LAB_DIR, 'select_action/final_robot_1st.csv'), help="CSV文件路径")
    parser.add_argument("--output_dir", type=str, default=os.path.join(_LAB_DIR, 'select_action'), help="输出目录")
    parser.add_argument("--num_frames", type=int, default=9, help="需要采样的帧数")
    parser.add_argument("--num_workers", type=int, default=128, help="Stage1的进程数")
    # 添加一个开关来控制是否进行可视化，因为这会增加I/O耗时
    parser.add_argument("--visualize", action="store_true", help="是否在Stage2生成可视化拼接图")
    # 添加一个采样率，例如每隔10个视频可视化一次，避免生成太多图片
    parser.add_argument("--vis_frequency", type=int, default=10000, help="可视化频率，1为全部可视化")

    args = parser.parse_args()

    # 路径准备
    cache_dir = os.path.join(args.output_dir, "human_diff_cache")
    vis_dir = os.path.join(args.output_dir, "human_sample_vis")
    os.makedirs(cache_dir, exist_ok=True)
    if args.visualize:
        os.makedirs(vis_dir, exist_ok=True)

    df = pd.read_csv(args.input_file)
    data_list = df.to_dict('records')

    print(f"Total videos to process: {len(data_list)}")
    print(f"Output directory: {args.output_dir}")
    if args.visualize:
        print(f"Visualization enabled (Frequency: every {args.vis_frequency} video)")

    # --- STAGE 1: 并行特征提取 ---
    print("\n=== Stage 1: Extracting SSIM Features (CPU/Disk Intensive) ===")
    with ProcessPoolExecutor(max_workers=args.num_workers) as executor:
        # 提交任务
        futures = [executor.submit(stage1_worker, row, cache_dir, args.num_frames*2) for row in data_list]
        # 使用 tqdm 监控进度
        for _ in tqdm(as_completed(futures), total=len(futures), desc="Extracting"):
            pass

    # --- STAGE 2: 采样与可视化 ---
    print("\n=== Stage 2: Generating Indices & Visualization (Single Process) ===")
    mg_sampler = MGSampler(clip_len=args.num_frames, test_mode=True)
    uni_sampler = UniformSampler(clip_len=args.num_frames)
    
    valid_samples = [] # 用于存放处理成功的行数据

    for i, row in enumerate(tqdm(data_list, desc="Sampling")):
        v_path = row['video_path']
        cache_path = get_safe_cache_path(v_path, cache_dir)
        
        try:
            # 1. 采样决策
            data = np.load(cache_path)
            if len(data) == 1 and data[0] == -1.0: # 视频过短标记
                cap_t = cv2.VideoCapture(v_path)
                total_f = int(cap_t.get(cv2.CAP_PROP_FRAME_COUNT))
                cap_t.release()
                current_indices = uni_sampler.sample({'filename': v_path, 'img_diff': [0]*total_f})
            else:
                current_indices = mg_sampler.sample({'filename': v_path, 'img_diff': data.tolist()})

            # 2. 去重与排序
            processed_indices = sorted(list(set([int(idx) for idx in current_indices])))

            # 3. [核心修改] 检查是否为空
            if not processed_indices or processed_indices == [0]:
                print(f"\n[Empty Result] Skipping sample (no indices generated): {v_path}")
                continue # 跳过当前循环，不保存该样本

            # 4. 长度不足补齐，长度超出截断
            while len(processed_indices) < args.num_frames:
                processed_indices.append(processed_indices[-1])
            
            final_indices = processed_indices[:args.num_frames]
            
            # 5. 可视化 (可选)
            if args.visualize and (i % args.vis_frequency == 0):
                save_visualization_jump(v_path, final_indices, row.get('action', 'NA'), vis_dir)

            # 6. 将处理好的索引存回 row，并加入有效样本列表
            row['sample_indices'] = ",".join(map(str, final_indices))
            valid_samples.append(row)

        except Exception as e:
            print(f"\n[Error] Skipping {v_path} due to: {e}")

    # --- 最终保存 ---
    # 使用有效样本列表创建新的 DataFrame，确保 sample_indices 和数据行对齐
    result_df = pd.DataFrame(valid_samples)
    out_csv = os.path.join(args.output_dir, "sample_human_1st.csv")
    result_df.to_csv(out_csv, index=False)
    
    print(f"\nProcessing complete!")
    print(f"Original samples: {len(df)}")
    print(f"Valid samples saved: {len(result_df)}")
    print(f"Total skipped: {len(df) - len(result_df)}")
    if args.visualize:
        print(f"Visualizations saved to: {vis_dir}")

if __name__ == "__main__":
    main()