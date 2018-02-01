# fill mask with OpenCV inpainting (Navier-Stokes)

from argparse import ArgumentParser
import cv2

parser = ArgumentParser()
parser.add_argument("--image", action="store", dest="image", metavar="IMAGE", help="image to be inpainted", required=True)
parser.add_argument("--mask", action="store", dest="mask", metavar="MASK", help="mask to apply", required=True)
parser.add_argument("--output", action="store", dest="output", metavar="OUTPUT", help="output path", required=True)


args = parser.parse_args()

def inpaint(image, mask, out):
    img = cv2.imread(image)
    mask = cv2.imread(mask,0)
    dst = cv2.inpaint(img,mask,3,cv2.INPAINT_TELEA)
    cv2.imshow('dst',dst)
    cv2.imwrite(out, dst)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# run it!
inpaint(args.image, args.mask, args.output)



