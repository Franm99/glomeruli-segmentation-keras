import os
from tqdm import tqdm
from skimage import io
from PIL import Image
from mask_workflow import workflow_test

patch_size = 256
# TODO: Specify path to my own dataset (D:/)

def get_subpatches_glomeruli(data_path: str):
    dataNames = [i for i in os.listdir(data_path) if i.endswith('.png')]
    patch_size_or = patch_size * 5
    for dataName in tqdm(dataNames):
        im = io.imread(data_path + dataName)
        dirName = data_path + dataName[:-4] + '/'
        if not os.path.isdir(dirName):
            os.mkdir(dirName)

        [h, w, _] = im.shape
        patch_num = 0
        for x in range(0, w, patch_size_or):
            if x+patch_size_or >= w:
                x = w - patch_size_or
            for y in range(0, h, patch_size_or):
                if y+patch_size_or >= h:
                    y = h - patch_size_or
                patch = Image.fromarray(im[y:y+patch_size_or, x:x+patch_size_or, :])
                patch = patch.resize((patch_size, patch_size))
                patch_name = str(patch_num).rjust(6, '0') + '.png'
                if not os.path.isfile(dirName + patch_name):
                    patch.save(dirName + patch_name)
                patch_num += 1


def run_workflow_test():
    ims_path = "myImages/"
    weights_filename = 'mitochondria_test.hdf5'
    workflow_test(weights_filename, ims_path)


if __name__ == '__main__':
    run_workflow_test()
