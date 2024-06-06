import numpy as np
from PIL import Image

if __name__ == "__main__":
    img_path = '/home/juan/Documents/tcc/examples/orto_ebee_BONF-SOCAMAI83-36028z111_13-10-2023_13-10-2023_14-10-2023_3968_32640patch_192_64.png'

    img = np.array(Image.open(img_path))
    np.set_printoptions(threshold=np.inf)
    print(img.shape)
    print(img)
