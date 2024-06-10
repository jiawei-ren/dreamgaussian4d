import os
import cv2

os.makedirs('data/CONSISTENT4D_DATA/test_dataset/eval_gt_rgb', exist_ok=True)
file_list = []
for img_name in ['aurorus', 'crocodile', 'guppie', 'monster', 'pistol', 'skull', 'trump']:
    os.makedirs(f'data/CONSISTENT4D_DATA/test_dataset/eval_gt_rgb/{img_name}', exist_ok=True)
    for view in range(4):
        os.makedirs(f'data/CONSISTENT4D_DATA/test_dataset/eval_gt_rgb/{img_name}/eval_{view}', exist_ok=True)
        for t in range(32):
            file_list.append(f'data/CONSISTENT4D_DATA/test_dataset/eval_gt/{img_name}/eval_{view}/{t}.png')
for file in file_list:
    img = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    input_mask = img[..., 3:]
    input_mask = input_mask / 255.
    input_img = img[..., :3] * input_mask + (1 - input_mask) * 255
    fpath = file.replace('eval_gt', 'eval_gt_rgb')
    cv2.imwrite(fpath, input_img) 