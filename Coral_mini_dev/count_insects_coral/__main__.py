import count_insects_coral.init as init
import count_insects_coral.camera as cam
import count_insects_coral.myhttp as myhttp
import count_insects_coral.interpreter_tf as interp
import count_insects_coral.interpreter_tf_tpu as interp_tpu

url_download='https://firmware.insectronics.net/giannis/'
filename_download='model_count_bugs_final_quant.tflite'

device_id='ad803f444d93f80d'

url_server_settings='http://insdev.trp.gr/settings/'
model='model_count_insects_final_2_quant.tflite'
model_tpu='model_count_insects_final_2_quant_edgetpu.tflite'

url_server_upload='http://insdev.trp.gr/binaries/ad803f444d93f80d'

if __name__ == "__main__":
    url=''
    init.my_logs.info('main')
    image=cam.capture_image()
    #interp.init_interpreter_tf(model)
    interp_tpu.init_interpreter_tf(model_tpu)

    #predict=interp.interpreter_evaluation_tf(image)
    
    predict2=interp_tpu.interpreter_evaluation_tf(image)
    
    filename=cam.create_name_of_image(100,predict2)
    cam.save_image(filename,image)
    #myhttp.download_tflite(url_download,filename_download)
    myhttp.send_image(url_server_upload,filename)
