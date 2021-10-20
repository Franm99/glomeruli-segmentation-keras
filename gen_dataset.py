from tqdm import tqdm
import os
from skimage import io
from PIL import Image

patch_size = 256


def get_images_from_tif(data_path: str):
    dataNames = [i for i in os.listdir(data_path) if i.endswith('.tif')]
    for dataName in tqdm(dataNames):
        ims = io.imread(data_path + dataName)
        dirName = data_path + dataName[:-4] + '/'
        if not os.path.isdir(dirName):
            os.mkdir(dirName)

        [num_ims, h, w] = ims.shape
        patch_num = 0
        for i in tqdm(range(num_ims), leave=False):
            im = ims[i]
            stride = 100
            for x in range(0, w, patch_size-stride):
                if x+patch_size >= w:
                    continue
                for y in range(0, h, patch_size-stride):
                    if y+patch_size >= h:
                        continue
                    patch = Image.fromarray(im[y:y+patch_size, x:x+patch_size])
                    patch_name = str(patch_num).rjust(6, '0') + '.png'
                    if not os.path.isfile(dirName + patch_name):
                        patch.save(dirName + patch_name)
                    patch_num += 1

