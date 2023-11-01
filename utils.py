import heapq
import torch
import os
class TopModelHeap:
    def __init__(self, output_dir, max_size=10):
        self.max_size = max_size
        self.models = []  # 使用堆来保存模型性能
        self.current_size = 0
        self.output_dir = output_dir

    def push(self, performance,model_name, state_dict):
        # 将模型名称和性能添加到堆中
        # 如果堆大小超过最大值，将弹出性能最低的模型
        if self.current_size < self.max_size:
            heapq.heappush(self.models, (-performance, model_name,state_dict))
            self.save_models_state_dicts(model_name,state_dict)
            self.current_size += 1
        else:
            heapq.heappush(self.models, (-performance, model_name, state_dict))
            _, old_model_name, old_state_dict = heapq.heappop(self.models)  # 弹出性能最低的模型
            # 删除对应的state_dict文件
            if old_model_name != model_name:
                self.save_models_state_dicts(model_name,state_dict)
            self.delete_state_dict_file(old_model_name)
        self.save_to_txt(os.path.join(self.output_dir,'best_recoder.txt'))
    
    def delete_state_dict_file(self, model_name):
        model_path = os.path.join(self.output_dir,model_name)
        # 删除state_dict文件
        if os.path.exists(model_path):
            os.remove(model_path)

    def get_top_models(self):
        # 返回性能最高的模型列表
        top_models = [(model_name, -performance) for performance, model_name,state_dict in sorted(self.models)]
        return top_models

    def save_to_txt(self, filename):
        # 将性能最高的模型记录到文本文件
        top_models = self.get_top_models()
        with open(filename, 'w') as file:
            for model_name, performance in top_models:
                file.write(f"{model_name}: {-performance}\n")
    
                
    def save_models_state_dicts(self, model_name,state_dict):
        # 保存性能最高的模型的state_dict到指定目录
        model_path = os.path.join(self.output_dir,model_name)
        torch.save(state_dict, model_path)
                