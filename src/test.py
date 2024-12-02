import os.path
import pickle
name = os.path.join("./evaluator_data/",\
                                "evaluate_net_dataset_cpu0_0")

datasets = []
data_path = "./evaluator_data/"
for idx,file in enumerate(os.listdir(data_path)):
    filename = os.path.join(data_path,file)
    with open(filename, 'rb') as fo:
        datasets.extend(pickle.load(fo, encoding='bytes'))

print(datasets)