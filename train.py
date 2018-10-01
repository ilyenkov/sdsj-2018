import argparse
import os
import pandas as pd
import numpy as np
import pickle
import time
import copy
import gc

from utils import transform_datetime_features, make_dtype

# use this to stop the algorithm before time limit exceeds
TIME_LIMIT = int(os.environ.get('TIME_LIMIT', 5*60))
timereserve = 15

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train-csv')#, type=argparse.FileType('r'), required=True)
    parser.add_argument('--model-dir', required=True)
    parser.add_argument('--mode', choices=['classification', 'regression'], required=True)
    args = parser.parse_args()

    start_time = time.time()
    print('TIME LIMIT:', TIME_LIMIT)
    with open('log.txt', 'a') as f:
        f.write(str(time.ctime()) + '\n')
    time_flag = False
    print(args.train_csv)
    
    dtypes=make_dtype(args.train_csv)
    print(len(dtypes))
    df = pd.read_csv(args.train_csv, dtype=dtypes)
    df_y = df.target
    df_X = df.drop('target', axis=1)

    print('Dataset read, shape {}'.format(df_X.shape))

    # dict with data necessary to make predictions
    model_config = {}
    
    #cols = df_X.columns.tolist()
    #strs = [c for c in cols if c.startswith('string')]
    #datas = [c for c in cols if c.startswith('datetime')]
    #nums = [c for c in cols if c.startswith('number')]    
    #time.sleep(0.1*len(strs))
    time.sleep(TIME_LIMIT - (time.time() - start_time) - 60)

    #time.sleep(0.0001*df.shape[0])
    #time.sleep(0.1*df.shape[1])

    model_config_filename = os.path.join(args.model_dir, 'model_config.pkl')
    with open(model_config_filename, 'wb') as fout:
        pickle.dump(model_config, fout, protocol=pickle.HIGHEST_PROTOCOL)
        
    print('Train time: {}'.format(time.time() - start_time))
    