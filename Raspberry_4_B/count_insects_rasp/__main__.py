import count_insects_rasp.init as init
import count_insects_rasp.camera as cam
import count_insects_rasp.myhttp as myhttp
import count_insects_rasp.interpreter as interp

url_download='https://firmware.insectronics.net/giannis/'
filename_download='model_count_insects_final_2_quant.tflite'

device_id='ad803f444d93f80d'

url_server_settings='http://insdev.trp.gr/settings/'
model='model_count_insects_final_2_quant.tflite'

url_server_upload='http://insdev.trp.gr/binaries/ad803f444d93f80d'

if __name__ == "__main__":
    url=''
    init.my_logs.info('main')
    image=cam.capture_image()
    interp.init_interpreter(model)
    predict=interp.interpreter_evaluation(image)
    filename=cam.create_name_of_image(100,predict)
    cam.save_image(filename,image)
    #myhttp.download_tflite(url_download,filename_download)
    myhttp.send_image(url_server_upload,filename)
