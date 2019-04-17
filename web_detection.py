import base64
import json
import os

from googleapiclient import discovery
from oauth2client.client import GoogleCredentials

os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "secrets.json"


image_path = "malala.jpg"

def main():

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
                    'type': 'WEB_DETECTION',
                    'maxResults': 10
                }]
            }]
        })
        response = service_request.execute()
        print(json.dumps(response, indent=4, sort_keys=True))

if __name__ == '__main__':
    main()
