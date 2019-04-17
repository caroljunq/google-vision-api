# https://cloud.google.com/vision/docs/detecting-safe-search
from google.cloud import vision
from google.cloud.vision import types
from PIL import Image, ImageDraw
import cv2
import io

import os

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="secrets.json"

client = vision.ImageAnnotatorClient()

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

def detect_safe_search(path):
    """Detects unsafe features in the file."""
    from google.cloud import vision
    client = vision.ImageAnnotatorClient()

    with io.open(path, 'rb') as image_file:
        content = image_file.read()

    image = vision.types.Image(content=content)

    response = client.safe_search_detection(image=image)
    safe = response.safe_search_annotation

    # Names of likelihood from google.cloud.vision.enums
    likelihood_name = ('UNKNOWN', 'VERY_UNLIKELY', 'UNLIKELY', 'POSSIBLE',
                       'LIKELY', 'VERY_LIKELY')
    print('Safe search:')

    print('adult: {}'.format(likelihood_name[safe.adult]))
    print('medical: {}'.format(likelihood_name[safe.medical]))
    print('spoofed: {}'.format(likelihood_name[safe.spoof]))
    print('violence: {}'.format(likelihood_name[safe.violence]))
    print('racy: {}'.format(likelihood_name[safe.racy]))

def main():
    take_picture()
    detect_safe_search(image_path)

if __name__ == '__main__':
    main()