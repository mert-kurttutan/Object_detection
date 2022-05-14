import logging
import sys
import argparse
from pipeline_run import preprocess
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
    
    parser.add_argument('--blob-path',
                          type=str,
                          help='Zip file where the preprocessed data is stored',
                          default='training_data.zip')
    
    parser.add_argument('--kfp',
                          dest='kfp',
                          action='store_true',
                          help='Kubeflow pipelines flag')
    
    parser.add_argument('--output1-path', type=str,
          help='Path of the local file where the Output 1 data should be written.')

    args, _ = parser.parse_known_args(args=argv[1:])

    return args




def run_preprocess(argv=None):
    """Transforms data"""
    
    args = parse_arguments(sys.argv if argv is None else argv)
    preprocess(bucket=args.bucket, data_zip=args.blob_path)
    
    if args.kfp:
        # Creating the directory where the output file is created (the directory
        # may or may not exist).
        Path(args.output1_path).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output1_path).write_text(args.blob_path)
    
if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    run_preprocess()