import os

dataset_base = '/gpfs/share/home/1701214057/CUB2011/datasets/CUB_200_2011'

image_from_id_to_path = dict()

with open(os.path.join(dataset_base, 'images.txt'), 'r') as f:
    for line in f.readlines():
        image_id, image_name = line.strip().split(' ')
	image_path = os.path.join(dataset_base, 'images', image_name)
        image_from_id_to_path[image_id] = image_path


image_from_id_to_train_or_test = dict()
with open(os.path.join(dataset_base, 'train_test_split.txt'), 'r') as f:
    for line in f.readlines():
        image_id, label = line.strip().split(' ')
        image_from_id_to_train_or_test[image_id] = label

train_data_info = []
test_data_info = []
for key, val in image_from_id_to_path.items():
    label = image_from_id_to_train_or_test[key]
    class_id = int(val.split('/')[-2].split('.')[0]) - 1
    if label == '1':
        train_data_info.append((val, class_id))
    elif label == '0':
        test_data_info.append((val, class_id))

with open('train_data_info.txt', 'w') as f:
    for image_path, class_id in train_data_info:
        print(image_path, class_id)
        f.write('{} {}\n'.format(image_path, str(class_id)))

with open('test_data_info.txt', 'w') as f:
    for image_path, class_id in test_data_info:
        print(image_path, class_id)
        f.write('{} {}\n'.format(image_path, str(class_id)))

