import streamlit as st
import cv2
import numpy as np
import requests
import tempfile



file=open('D:/DataScience/deep learning/objects_list.txt')
li=file.read().split('\n')
classes=list(map(str.strip,li))
file.close()

cfg_url='https://raw.githubusercontent.com/AlexeyAB/darknet/master/cfg/yolov4.cfg'
weights_url='https://github.com/AlexeyAB/darknet/releases/download/darknet_yolo_v3_optimal/yolov4.weights'

#funtion to download file into a temporary file
def download_to_tempfile(url):
    res=requests.get(url)
    res.raise_for_status()  # raise error if request failed
    tem=tempfile.NamedTemporaryFile(delete=False)
    tem.write(res.content)
    tem.close()
    return tem.name

#Download cfg and weights
cfg_file=download_to_tempfile(cfg_url)
weights_file=download_to_tempfile(weights_url)

model=cv2.dnn.readNetFromDarknet(cfg_file,weights_file)

detection_model = cv2.dnn_DetectionModel(model)
detection_model.setInputSize(416, 416)
detection_model.setInputScale(1/255)



def detect(path):
    np_img=np.frombuffer(path,np.uint8)
    img=cv2.imdecode(np_img,cv2.IMREAD_COLOR)

    
    classIds,classProbs,bboxes=detection_model.detect(img,confThreshold=.75,nmsThreshold=.5)
  
    for box,cls,prob in zip(bboxes,classIds,classProbs):
        x,y,w,h=box
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),4)
        cv2.putText(img,f'{classes[cls]}({prob:.2f})',(x,y-10),cv2.FONT_HERSHEY_PLAIN,1,(0,0,0),2)

    return img

# Page title
st.header('Object Detection')

# Add custom CSS
st.markdown("""
    <style>
    .banner {
        background-color: #4CAF50; /* Green background */
        color: white; /* White text */
        padding: 20px;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
        border-radius: 8px; /* Optional rounded corners */
        margin-bottom: 20px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Use the banner div
st.markdown('<div class="banner">Welcome to the Object Detection App 🚀</div>', unsafe_allow_html=True)

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["png","jpg","jfif"])

col1,col2=st.columns(2)
if uploaded_file:
    with col1:
        # Display original image
        st.image(uploaded_file,caption="uploaded image",use_container_width=True)

    if st.button('Prediction'):
        # Read file bytes and run detection
        path=uploaded_file.read()
        img=detect(path)
       
        # Convert BGR to RGB for Streamlit display
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)    
        
        with col2:
            st.image(img_rgb, caption="Detection Result", use_container_width=True)
    

