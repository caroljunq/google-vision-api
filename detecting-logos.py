# Based on https://github.com/AhirtonLopes/Google-Recognition/blob/master/logo-recognition.py

import base64
import json
import os
import cv2

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secrets.json"


image_path = "image.png"


def take_picture():

    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Take a picture")

    while True:
        ret, frame = cam.read()
        cv2.imshow("test", frame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k%256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k%256 == 32:
            # SPACE pressed
            cv2.imwrite(image_path, frame)
            break
    cam.release()

def main():
    take_picture()

    credentials = GoogleCredentials.get_application_default()
    service = discovery.build('vision', 'v1', credentials=credentials)

    with open(image_path, 'rb') as image:
        image_content = base64.b64encode(image.read())
        service_request = service.images().annotate(body={
            'requests': [{
                'image': {
                    'content': image_content.decode('UTF-8')
                },
                'features': [{
                    'type': 'LOGO_DETECTION',
                    'maxResults': 100
                }]
            }]
        })
        response = service_request.execute()
        
        try:
             label = response['responses'][0]['logoAnnotations'][0]['description']
        except:
             label = "No response."
        
        print(label)

if __name__ == '__main__':
    main()
