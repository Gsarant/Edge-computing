import logging


def init_logs(filename):
    try:
        for h in logging.getLogger().handlers:
            logging.getLogger().removeHandler(h)
    except:
        pass
    my_logs = logging.getLogger(filename)
    my_logs.setLevel(logging.INFO)
    # Handler - 1
    h_file = logging.FileHandler(f'{filename}.log')
    fileformat = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
    h_file.setLevel(logging.INFO)
    h_file.setFormatter(fileformat)

    h_stream = logging.StreamHandler()
    h_streamformat = logging.Formatter("%(asctime)s:   %(message)s")
    h_stream.setLevel(logging.INFO)
    h_stream.setFormatter(h_streamformat)

    # Adding all handlers to the logs
    my_logs.addHandler(h_file)
    my_logs.addHandler(h_stream)
    
    my_logs.info('Init Logs')
