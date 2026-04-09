import numpy as np
import random

class UniformSampler():
    """Uniformly sample frames from the video.
    Args:
        clip_len (int): Number of frames in each sampled output clip.
    """

    def __init__(self, clip_len=16):
        self.clip_len = clip_len

    def sample(self, results):
        """Sample frames uniformly from the video.
        Args:
            results (dict): Dictionary containing video information including
                            the number of frames.
        Returns:
            list: List of indices representing the sampled frames.
        """
        num_frames = len(results['img_diff'])
        
        # Calculate the interval for uniform sampling
        interval = num_frames / self.clip_len
        
        # Sample indices uniformly
        choose_index = [int(interval * i) for i in range(self.clip_len)]
        
        return choose_index
    
class MGSampler():
    """Sample frames from the video.
    Args:
        clip_len (int): Frames of each sampled output clip.
    """

    def __init__(self,
                 clip_len=16,
                 test_mode=True):

        self.clip_len = clip_len
        self.test_mode = test_mode

    def sample(self, results):

        def find_nearest(array, value):
            array = np.asarray(array)
            try:
                idx = (np.abs(array - value)).argmin()
                return int(idx + 1)
            except(ValueError):
                print(results['filename'])

        diff_score = results['img_diff']
        diff_score = np.power(diff_score, 0.5)
        sum_num = np.sum(diff_score)
        diff_score = diff_score / sum_num

        count = 0
        pic_diff = list()
        for i in range(len(diff_score)):
            count = count + diff_score[i]
            pic_diff.append(count)

        choose_index = list()

        if self.test_mode:
            for i in range(self.clip_len):
                choose_index.append(find_nearest(pic_diff, 1 / (self.clip_len * 2) + i / self.clip_len))
        else:
            for i in range(self.clip_len):
                choose_index.append(find_nearest(pic_diff, random.uniform(i / self.clip_len, (i + 1) / self.clip_len)))

        return choose_index