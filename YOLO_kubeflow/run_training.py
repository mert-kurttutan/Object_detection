import logging
import sys
import argparse
from pipeline_run import train
from pathlib import Path

def parse_arguments(argv=None):
    """Parse command line arguments
    Args:
      argv (list): list of command line arguments including program name
    Returns:
      The parsed arguments as returned by argparse.ArgumentParser
    """
    parser = argparse.ArgumentParser(description='Preprocessing')

    parser.add_argument('--bucket',
                          type=str,
                          help='GCS bucket where preprocessed data is saved',
                          default='obj-det-kb123')
    
    parser.add_argument('--kfp',
                          dest='kfp',
                          action='store_true',
                          help='Kubeflow pipelines flag')
    
    parser.add_argument('--blob-path',
                          type=str,
                          help='GCS blob path where data is saved',
                          default='training_data.zip')

    args, _ = parser.parse_known_args(args=argv[1:])

    return args

def run_training(argv=None):
    """Transforms data"""
    
    args = parse_arguments(sys.argv if argv is None else argv)
    f = open(args.blob_path, "r")
    train(bucket=args.bucket, data_zip=f.read())
    

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_training()