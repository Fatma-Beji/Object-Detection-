import cv2
import pickle


model = pickle.load(open("model.pkl","rb"))

classNames = []
classFile='coco.names'
with open(classFile,'rt') as f:
    classNames=[line.rstrip() for line in f]

configPath = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
weightsPath = 'frozen_inference_graph.pb'

net = cv2.dnn_DetectionModel(weightsPath,configPath)
net.setInputSize(320, 320)
net.setInputScale(1.0 / 127.5)
net.setInputMean((127.5, 127.5, 127.5))
net.setInputSwapRB(True)

def most_frequent(List):
    counter = 0
    num = List[0]

    for i in List:
        curr_frequency = List.count(i)
        if(curr_frequency> counter):
            counter = curr_frequency
            num = i
 
    return num
freq = 100
class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        self.video.set(3,640)
        self.video.set(4,480)

    def __del__(self):
        self.video.release()
    def get_frame(self):

        ret, frame = self.video.read()
        classIds, confs, bbox = net.detect(frame, confThreshold=0.5)
        print(classIds,bbox)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        height, width, _ = frame.shape

        if len(classIds) != 0:
            for classId, confidence,box in zip(classIds.flatten(),confs.flatten(),bbox):
                cx = int(box[0] + box[2] / 2)
                cy = int(box[1] + box[3] / 2)
                ccolors=[]
                for x in range(box[0], box[2]):
                #for y in range(box[1], box[3]):
                    #print(rgb_frame[cy][cx])
                    r = rgb_frame[cy][x][0]
                    g = rgb_frame[cy][x][1]
                    b = rgb_frame[cy][x][2]
                    res=model.predict([[r,g,b]])
                    ccolors.append(res[0])
                if len(ccolors): 
                    global freq
                    freq=most_frequent(ccolors)
                    print(freq)
                if freq == 0 :
                    freqq = "Black"
                elif freq == 1 :
                    freqq = "Blue"
                elif freq ==  2 :
                    freqq = "Brown"
                elif freq == 3 :
                    freqq = "Green"
                elif freq == 4 :
                    freqq = "Orange"
                elif freq == 5 :
                    freqq = "Pink"
                elif freq == 6 :
                    freqq = "Purple"
                elif freq == 7 :
                    freqq = "Red"
                elif freq == 8 :
                    freqq = "White"
                else:
                    freqq = "Yellow"
                print(freqq)
                cv2.rectangle(frame,box,color=(0,0,0),thickness=2)
                cv2.putText(frame,classNames[classId-1].upper(),(box[0]+10,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
                cv2.putText(frame,freqq,(box[0]+200,box[1]+30),
                    cv2.FONT_HERSHEY_COMPLEX,1,(0,0,0),2)
                #cv2.putText(img,str(round(confidence*100,2)),(box[0]+200,box[1]+30),
                            #cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),2)

        ret, jpg = cv2.imencode('.jpg', frame)
        return jpg.tobytes()
