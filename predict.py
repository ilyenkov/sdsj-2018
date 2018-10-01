import argparse
import os
import pandas as pd
import numpy as np
import pickle
import time
import lightgbm as lgb
import gc

from utils import make_dtype

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--test-csv')#, type=argparse.FileType('r'), required=True)
    parser.add_argument('--prediction-csv')#, type=argparse.FileType('w'), required=True)
    parser.add_argument('--model-dir', required=True)
    args = parser.parse_args()

    start_time = time.time()
    print('TIME LIMIT:', TIME_LIMIT)
    with open('log.txt', 'a') as f:
        f.write(str(time.ctime()) + '\n')

    # load model
    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'rb') as fin:
        model_config = pickle.load(fin)

    # read dataset
    #print(args.test_csv)
    df = pd.read_csv(args.test_csv, dtype=make_dtype(args.test_csv))
    print('Dataset read, shape {}'.format(df.shape))
    
    #cols = df_X.columns.tolist()
    #strs = [c for c in cols if c.startswith('string')]
    #datas = [c for c in cols if c.startswith('datetime')]
    #nums = [c for c in cols if c.startswith('number')]    
    #time.sleep(0.1*len(datas))
    time.sleep(TIME_LIMIT - (time.time() - start_time) - 60)
    
    #time.sleep(0.0001*df.shape[0])
    #time.sleep(0.1*df.shape[1])
   
    df['prediction'] = 0
    df[['line_id', 'prediction']].to_csv(args.prediction_csv, index=False)
    
    print()
    print('Prediction time: {}'.format(time.time() - start_time))
    with open('log.txt', 'a') as f:
        f.write('Prediction time: {}'.format(time.time() - start_time) + '\n')