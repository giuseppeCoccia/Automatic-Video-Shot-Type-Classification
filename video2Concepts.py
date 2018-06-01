# Retrieve module:
# It is used for extracting the retrieval images from CNNs feature vector comparison
import numpy as np
import os, sys, getopt
import cv2
import imageio
from skimage import img_as_ubyte
import skimage
import time
# Main path to your caffe installation
my_caffe_root = '/datas/teaching/projects/fall2015/pf27/System/'
caffe_root = "/opt/caffe/" 

# Model prototxt file
#model_prototxt = my_caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
#model_prototxt = my_caffe_root + 'models/resnet/ResNet-152-deploy.prototxt'
model_prototxt = my_caffe_root + 'models/bvlc_googlenet/deploy.prototxt'

# Model caffemodel file
#model_trained = my_caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'
#model_trained = my_caffe_root + 'models/resnet/ResNet-152-model.caffemodel'
model_trained = my_caffe_root + 'models/bvlc_googlenet/bvlc_googlenet.caffemodel'

# File containing the class labels
imagenet_labels = my_caffe_root + 'data/ilsvrc12/synset_words.txt'
 
# Path to the mean image (used for input processing)
mean_path = my_caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy'
#mean_path = '/home/truong-an/Downloads/ilsvrc_2012_mean.npy'
# Name of the layer we want to extract
layer_name = 'fc1000'
 
#sys.path.insert(0, caffe_root + 'python')
import caffe
from copy import copy

def sortPredict(predicts):
    numero = len(predicts[0])-1
    numero2 = len(predicts[0])-2
    idxLs = numero * [None]
    predictLs = copy(predicts)
    for z in range (0,numero):
	idxLs[z] = z
    while True:
	for j in range(0,numero2):
	    if predictLs[0][j] < predictLs[0][j+1]:
		temp = predictLs[0][j+1]
		predictLs[0][j+1] = predictLs[0][j]
		predictLs[0][j] = temp
		temp1 = idxLs[j+1]
		idxLs[j+1] = idxLs[j]
		idxLs[j] = temp1
	numero2 = numero2 - 1
	if numero2 == 0:
	    break
    return idxLs

 
def main(argv):
    inputfile = ''
    outputfile = ''
 
    try:
        opts, args = getopt.getopt(argv,"hi:o:",["ifile=","ofile="])
    except getopt.GetoptError:
        print 'video2Concepts -i <inputfile> -o <outputfile>'
        sys.exit(2)
 
    for opt, arg in opts:
        if opt == '-h':
            print 'video2Concepts.py -i <inputfile> -o <outputfile>'
            sys.exit()
        elif opt in ("-i"):
            inputfile = arg
        elif opt in ("-o"):
            outputfile = arg
			
	print 'Reading images from "', inputfile
	print 'Writing vectors to "', outputfile
	# Setting this to CPU, but feel free to use GPU if you have CUDA installed
    caffe.set_mode_gpu()
	# Loading the Caffe model, setting preprocessing parameters
    net = caffe.Classifier(model_prototxt, model_trained,
                           mean=np.load(mean_path).mean(1).mean(1),
                           channel_swap=(2,1,0),
                           raw_scale=255,
                           image_dims=(256, 256))
    # Loading class labels
    with open(imagenet_labels) as f:
 
        labels = f.readlines()
 
    # This prints information about the network layers (names and sizes)
    # You can uncomment this, to have a look inside the network and choose which layer to print
    #print [(k, v.data.shape) for k, v in net.blobs.items()]
    #exit()
    #start_time = time.time()
    #vid_out = cv2.VideoWriter(outputfile,cv2.cv.CV_FOURCC('M','P','4','V'),25,(wi+400,he))
   
    #vid = imageio.get_reader(inputfile,'ffmpeg')
	vid_in = cv2.VideoCapture(inputfile);
    #frame1 = vid.get_data(0)
	he = int(vid_in.get(cv2.cv.CV_CAP_PROP_FRAME_HEIGHT))
	wi = int(vid_in.get(cv2.cv.CV_CAP_PROP_FRAME_WIDTH))
	vid_frames = int(vid_in.get(cv2.cv.CV_CAP_PROP_FRAME_COUNT))
	vid_fps = 25
	#int(vid_in.get(cv2.cv.CV_CAP_PROP_FPS))
	print 'width=', wi, ' height=', he, ' #frames=', vid_frames, ' fps=', vid_fps
    #he, wi, la = frame1.shape
    #video.open(outputfile,cv2.cv.CV_FOURCC('M','P','4','V'),25,(wi+400,he))
	vid_out = cv2.VideoWriter(outputfile,cv2.cv.CV_FOURCC('M','P','4','V'),vid_fps,(wi+400,he),1)
	
	pred = [None]*5
	prob = [None]*5
    #if video.isOpened() ==  True:
	
	time_begin = time.time()
	frame = 0
    #for frame in range(0,vid_frames):
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
		
#		image = cv2.imread('test_vi1.png') 
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
	#cv2.imshow('Video Stream', image)
	vid_out.release()
	vid_in.release()
	
    elapsed_time = time.time() - time_begin
    print 'Elapsed Time ', elapsed_time

    # Processing one image at a time, printint predictions and writing the vector to a file
    #with open(inputfile, 'r') as reader:
    #    with open(outputfile, 'w') as writer:
    #        writer.truncate()
    #        for image_path in reader:
    #            image_path = image_path.strip()
    #            input_image1 = skimage.img_as_float(skimage.io.imread('test_beach.jpg')).astype(np.float32)#caffe.io.load_image('road.jpg')
	#	input_image = imageio.imread('test_beach.jpg')
	#		input_image = skimage.img_as_float(input_image).astype(np.float32)
    #            prediction = net.predict([input_image],oversample=True)
	#	pred = sortPredict(prediction)                
		#print os.path.basename(image_path), ' : ' , labels[prediction[0].argmax()].strip() , ' (', prediction[0][prediction[0].argmax()] , ')'
		# output top 5 probability prediction
	#	img_cv = img_as_ubyte(input_image)
	#	for y in range(0,5):
#		    print os.path.basename(image_path), ' : ' , labels[pred[y]].strip() , ' (', prediction[0][pred[y]] , ')', '\n' 
#		    a = labels[pred[y]].strip() + ' ('+ str(prediction[0][pred[y]]) + ')' ;
#		    font  = cv2.FONT_HERSHEY_SIMPLEX   		 
#		    cv2.putText(img_cv,a,(50,400+20*y),font,0.6,(0,0,0),2) 
#		cv2.imwrite("test_beach_write.png", img_cv)        
#		np.savetxt(writer, net.blobs[layer_name].data[0].reshape(1,-1), fmt='%.8g')
if __name__ == "__main__":
    main(sys.argv[1:])



#cap = cv2.VideoCapture('test_video.mp4')

#while(cap.isOpened()):
#    ret, frame = cap.read()

#    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

#    cv2.imshow('frame',gray)
#    if cv2.waitKey(1) & 0xFF == ord('q'):
#        break

#cap.release()
#cv2.destroyAllWindows()
