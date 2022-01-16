import requests
import os
import count_insects_coral.init as init
from datetime import datetime



def download_tflite(url,filename):
    r = requests.get(url+filename, allow_redirects=True)
    open(filename, 'wb').write(r.content)

def send_image(send_url,image_file_name):
    starttime=datetime.now()

    if os.path.isfile(image_file_name):
        if len(send_url)>0:
            files = {'file': open(image_file_name, 'rb')}
            rq = requests.post(send_url,files=files)
            init.my_logs.info (f' Send image {image_file_name}')

        else:
            init.my_logs.error(f'Error No URL to send')
    else:    
        init.my_logs.error(f'Error image send')
    endtime=datetime.now()
    init.my_logs.info(f'Upload time {(endtime-starttime).microseconds} microseconds')
