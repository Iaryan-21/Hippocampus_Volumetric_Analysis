import os
import json

from experiments.UNetExperiment import UNetExperiment
from data_prep.HippocampusDatasetLoader import Hippocampus_Data
from sklearn.model_selection import train_test_split

class config:
    def __init__(self):
        self.name = "UNet"
        self.root_dir = r""
        self.n_epochs = 100
        self.learning_rate = 0.0001
        self.batch_size = 8
        self.patch_size = 64
        self.test_results_dir = ""

if __name__ == "__main__":
    c = config()
    c.root_dir = r"C:\\Users\\aryan\\OneDrive\\Desktop\\Hypocampal Volume Quantification of Alzheimers\\model\\data\\"
    c.test_results_dir = r"./test_results"
    
    print("Loading Data")
    
    try:
        data = Hippocampus_Data(c.root_dir, y_shape=c.patch_size, z_shape=c.patch_size)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        exit(1)
    
    keys = range(len(data))
    split = dict()
    split['train'], val_test = train_test_split(keys, test_size=0.3, random_state=100)
    split['val'], split['test'] = train_test_split(val_test, test_size=0.5, random_state=100)
    
    print(len(split['train']), len(split['test']), len(split['val']))
    print(len(split['train']) / 260, len(split['test']) / 260, len(split['val']) / 260)
    
    exp = UNetExperiment(c, split, data)
    exp.run()
    
    results_json = exp.run_test()
    results_json["config"] = vars(c)

    with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
        json.dump(results_json, out_file, indent=2, separators=(',', ': '))

## --------------------------------------------------------------------------------- ##
'''
DEEPLLABV3+ Uncomment to USE:::
'''

# import os
# import json

# from experiments.Deeplabv3_plusExperiment import DeepLabV3PlusExperiment 
# from data_prep.HippocampusDatasetLoader import Hippocampus_Data
# from sklearn.model_selection import train_test_split

# class config:
#     def __init__(self):
#         self.name = "DeepLabV3Plus"
#         self.root_dir = r""
#         self.n_epochs = 100
#         self.learning_rate = 0.0001
#         self.batch_size = 8
#         self.patch_size = 64
#         self.test_results_dir = ""

# if __name__ == "__main__":
#     c = config()
#     c.root_dir = r"C:\\Users\\aryan\\OneDrive\\Desktop\\Hypocampal Volume Quantification of Alzheimers\\model\\data\\"
#     c.test_results_dir = r"./test_results"
    
#     print("Loading Data")
    
#     try:
#         data = Hippocampus_Data(c.root_dir, y_shape=c.patch_size, z_shape=c.patch_size)
#     except FileNotFoundError as e:
#         print(f"Error: {e}")
#         exit(1)
    
#     keys = range(len(data))
#     split = dict()
#     split['train'], val_test = train_test_split(keys, test_size=0.3, random_state=100)
#     split['val'], split['test'] = train_test_split(val_test, test_size=0.5, random_state=100)
    
#     print(len(split['train']), len(split['test']), len(split['val']))
#     print(len(split['train']) / 260, len(split['test']) / 260, len(split['val']) / 260)
    
#     exp = DeepLabV3PlusExperiment(c, split, data)
#     exp.run()
    
#     results_json = exp.run_test()
#     results_json["config"] = vars(c)

#     with open(os.path.join(exp.out_dir, "results.json"), 'w') as out_file:
#         json.dump(results_json, out_file, indent=2, separators=(',', ': '))
