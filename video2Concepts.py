import numpy as np
import os, sys, getopt
import cv2

inputfile = sys.argv[1]

vid_in = cv2.VideoCapture(inputfile)

he = int(vid_in.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
wi = int(vid_in.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
vid_frames = int(vid_in.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
vid_fps = 25
#int(vid_in.get(cv2.cv.CV_CAP_PROP_FPS))
print 'width=', wi, ' height=', he, ' #frames=', vid_frames, ' fps=', vid_fps

#video.open(outputfile,cv2.cv.CV_FOURCC('M','P','4','V'),25,(wi+400,he))
vid_out = cv2.VideoWriter(outputfile,cv2.cv.CV_FOURCC('M','P','4','V'),vid_fps,(wi+400,he),1)

frame = 0
while(vid_in.isOpened() & frame<vid_frames):
    ret, image = vid_in.read()
    if ret==True:
        #print frame, " ", ret
        if(frame % 5 == 0):
            #print '*'
            if(frame % (vid_fps*60) == 0):
                print frame, 's processed in ', time.time() - time_begin
            newres_image = cv2.resize(image,(227,227),interpolation = cv2.INTER_LINEAR)
            img = newres_image.astype(np.float32)/255.
            img = img[...,::-1]
            #cv2.imwrite('TMP_video2Concepts.png',newres_image)
            #inputs = [caffe.io.load_image('TMP_video2Concepts.png')]
            #prediction = net.predict([newres_image],oversample=True)
            prediction = net.predict([img], oversample=True)
            pred[0] = prediction[0].argmax()
            prob[0] = prediction[0][pred[0]]
            prediction[0][pred[0]] = 0
            pred[1] = prediction[0].argmax()
            prob[1] = prediction[0][pred[1]]
            prediction[0][pred[1]] = 0
            pred[2] = prediction[0].argmax()
            prob[2] = prediction[0][pred[2]]
            prediction[0][pred[2]] = 0
            pred[3] = prediction[0].argmax()
            prob[3] = prediction[0][pred[3]]
            prediction[0][pred[3]] = 0
            pred[4] = prediction[0].argmax()
            prob[4] = prediction[0][pred[4]]
            prediction[0][pred[4]] = 0
    
            l1 = labels[pred[0]].strip().split(' ')
            l2 = labels[pred[1]].strip().split(' ')
            l3 = labels[pred[2]].strip().split(' ')
            l4 = labels[pred[3]].strip().split(' ')
            l5 = labels[pred[4]].strip().split(' ')
    
            a = l1[1]+','+l1[len(l1)-1] + ' ('+ str(prob[0]) + ')'
            b = l2[1]+','+l2[len(l2)-1] + ' ('+ str(prob[1]) + ')'
            c = l3[1]+','+l3[len(l3)-1] + ' ('+ str(prob[2]) + ')'
            d = l4[1]+','+l4[len(l4)-1] + ' ('+ str(prob[3]) + ')'
            e = l5[1]+','+l5[len(l5)-1] + ' ('+ str(prob[4]) + ')'
    

        image = cv2.copyMakeBorder(image,0,0,0,400,cv2.BORDER_CONSTANT,value=0)
        font  = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image,a,(wi+10,100),font,0.6,(255,255,255),2)
        cv2.putText(image,b,(wi+10,120),font,0.6,(255,255,255),2)
        cv2.putText(image,c,(wi+10,140),font,0.6,(255,255,255),2)
        cv2.putText(image,d,(wi+10,160),font,0.6,(255,255,255),2)
        cv2.putText(image,e,(wi+10,180),font,0.6,(255,255,255),2)
        vid_out.write(image)
        frame = frame+1
    else:
        print "break reached!", ret
        break

vid_out.release()
vid_in.release()
