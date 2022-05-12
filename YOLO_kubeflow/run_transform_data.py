
import logging

from pipeline_run import transform_data



def run_data_loader():
    
    return transform_data()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    transform_data()