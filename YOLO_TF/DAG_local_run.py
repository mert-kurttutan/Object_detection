# /usr/bin/env python3

from absl import logging
# Local packages
import pipeline_run
from src import (dataParser as dp, 
                 dataLoader as dl,
                 obj_det_model,
                 utils)


logging.set_verbosity(logging.INFO)  
if __name__ == "__main__":
    logging.set_verbosity(logging.INFO) # Set default logging level.
    pipeline_run.run_pipeline()