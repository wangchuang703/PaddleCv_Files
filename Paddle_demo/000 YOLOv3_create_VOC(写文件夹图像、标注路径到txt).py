import os
# import random
import shutil

TRAIN_READ_DIR = 'sleeper_train/'
VALID_READ_DIR = 'sleeper_valid/'
WRITE_DIR = 'voc_sleeper/'
IMAGES_DIR = WRITE_DIR + 'images'
ANNOTATIONS_DIR = WRITE_DIR + 'annotations'

# # TRAIN_RATIO = 0.7
# # VAL_RATIO = 0.2
# # TEST_RATIO = 0.1

if not os.path.isdir(IMAGES_DIR):
    os.makedirs(IMAGES_DIR)
if not os.path.isdir(ANNOTATIONS_DIR):
    os.mkdir(ANNOTATIONS_DIR)

train_imgs = os.listdir(TRAIN_READ_DIR)
valid_imgs = os.listdir(VALID_READ_DIR)
train_annotations = os.listdir(TRAIN_READ_DIR+'outputs')
valid_annotations = os.listdir(VALID_READ_DIR+'outputs')

train_imgs.remove('outputs')
valid_imgs.remove('outputs')
# # Shuffle list imgs in place.
# # random.shuffle(imgs)  

with open(WRITE_DIR+'/train.txt', 'w') as f:
    info = ''
    for im in train_imgs:
        info += './images/'+im+' '
        info += './annotations/'+im[0:-4]+'.xml\n'
        shutil.copyfile(TRAIN_READ_DIR+im,WRITE_DIR+'images/'+im)
    f.write(info[0:-1])



with open(WRITE_DIR+'/valid.txt', 'w') as f:
    info = ''
    for im in valid_imgs:
        info += './images/'+im+' '
        info += './annotations/'+im[0:-4]+'.xml\n'
        shutil.copyfile(VALID_READ_DIR+im,WRITE_DIR+'images/'+im)
    f.write(info[0:-1])

for ant in train_annotations:
     shutil.copyfile(TRAIN_READ_DIR+'outputs/'+ant,WRITE_DIR+'annotations/'+ant)
for ant in valid_annotations:
     shutil.copyfile(VALID_READ_DIR+'outputs/'+ant,WRITE_DIR+'annotations/'+ant)

print("OK")