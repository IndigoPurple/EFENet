from skimage.measure import compare_ssim
from skimage.measure import compare_psnr
from vifp import vifp_mscale
import cv2
import numpy as np

def SSIM(img1, img2):
    return compare_ssim(img1, img2,multichannel=True)


def VIF(img1, img2):
    return vifp_mscale(img1, img2)


if __name__ == '__main__':

    img1 = cv2.imread('/fileserver/tanyang/projects/ref_sr_ytan/mutiscale_warping_train/denoise_result_mixed/12/sr.png')
    img2 = cv2.imread('/fileserver/tanyang/projects/ref_sr_ytan/mutiscale_warping_train/denoise_result_mixed/12/gt.png')

    img1 = np.array(img1 / 255.0, dtype=np.float32)
    img2 = np.array(img2 / 255.0, dtype=np.float32)
    print("test on same image SSIM:{}, VIF:{}".format(SSIM(img1, img2), VIF(img1, img2)))
    print (compare_psnr(img1,img2))
