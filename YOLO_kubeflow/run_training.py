
import logging

from pipeline_run import train

def run_training():
    
    return train()

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_training()