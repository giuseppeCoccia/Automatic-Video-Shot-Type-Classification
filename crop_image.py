import cv2
import sys

if(len(sys.argv) < 3):
    print("Usage: python script_name.py imagepath x1 y1 x2 y2")
    print("Usage: python script_name.py imagepath pixel_to_cut")
    exit()

# read the image
img_name = sys.argv[1]
img = cv2.imread(img_name)


if(len(sys.argv) == 3):
    len_y = len(img)
    len_x = len(img[0])

    pixels = int(sys.argv[2])

    # crop bottom
    img1 = img[0:len_y-pixels, 0:len_x]

    # crop up
    img2 = img[pixels:len_y, 0:len_x]

    # crop left
    img3 = img[0:len_y, pixels:len_x]

    # crop right
    img4 = img[0:len_y, 0:len_x-pixels]

    #cv2.imshow("bottom", img1)
    #cv2.waitKey(0)

    #cv2.imshow("up", img2)
    #cv2.waitKey(0)

    #cv2.imshow("left", img3)
    #cv2.waitKey(0)

    #cv2.imshow("right", img4)
    #cv2.waitKey(0)

    cv2.imwrite('bottom_'+img_name, img1)
    cv2.imwrite('up_'+img_name, img2)
    cv2.imwrite('left_'+img_name, img3)
    cv2.imwrite('right_'+img_name, img4)



else:
    x = int(sys.argv[2])
    y = int(sys.argv[3])
    w = int(sys.argv[4])-x
    h = int(sys.argv[5])-y

    crop_img = img[y:y+h, x:x+w]
    #cv2.imshow("cropped", crop_img)
    #cv2.waitKey(0)

    cv2.imwrite('cropped_'+img_name, crop_img)



