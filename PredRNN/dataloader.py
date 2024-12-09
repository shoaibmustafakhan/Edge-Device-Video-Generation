import os
import numpy as np
import logging

class DataLoader:
    def __init__(self, data_dir="processed_data"):
        self.action_classes = ["Biking", "JumpingJack", "Bowling"]
        self.logger = logging.getLogger(__name__)
        self.class_video_indices = {cls: 0 for cls in self.action_classes}
        self.current_class_index = 0
        self.data_dir = data_dir
        
    def get_video_paths(self, class_name):
        class_path = os.path.join(self.data_dir, class_name)
        return sorted([os.path.join(class_path, f) for f in os.listdir(class_path) if f.endswith('.npy')])
    
    def next_video(self):
        current_class = self.action_classes[self.current_class_index]
        video_paths = self.get_video_paths(current_class)
        
        if not video_paths:
            self.current_class_index = (self.current_class_index + 1) % len(self.action_classes)
            if self.current_class_index == 0:
                return None, None
            return self.next_video()
        
        current_video_index = self.class_video_indices[current_class]
        if current_video_index >= len(video_paths):
            self.class_video_indices[current_class] = 0
            self.current_class_index = (self.current_class_index + 1) % len(self.action_classes)
            if self.current_class_index == 0:
                return None, None
            return self.next_video()
        
        frames = np.load(video_paths[current_video_index])
        self.class_video_indices[current_class] += 1
        
        return frames, current_class