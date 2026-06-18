""" 
    Function:   Define some optional arguments and configurations
"""

import argparse
import time
import os

class opt():
    def __init__(self):
        time_stamp = time.time()
        local_time = time.localtime(time_stamp)
        self.time = time.strftime('%m%d%H%M', local_time)
        
    def parse(self):
        """ Function: Define optional arguments
        """
        parser = argparse.ArgumentParser(description='Self-supervised learing for multi-channel audio processing')
        parser.add_argument('--train', action='store_true', default=False, help='change to train stage (default: False)')
        parser.add_argument('--test', action='store_true', default=False, help='change to test stage (default: False)')
        parser.add_argument('--dev', action='store_true', default=False, help='change to test stage (default: False)')
        # for both training and test stages
        parser.add_argument('--gpu-id', type=int, default=0, metavar='GPU', help='GPU ID (default: 7)')
        parser.add_argument('--sources', type=int, nargs='+', default=[1,2], metavar='Sources', help='Number of sources (default: 1)')
        parser.add_argument('--source-state', type=str, default='mobile', metavar='SourceState', help='State of sources (default: Mobile)')    
        args = parser.parse_args()

        return args
        
    def dir(self):
        """ Function: Get directories of code, data and experimental results
        """ 
        dirs = {}

        dirs['data'] = "/mnt/e/"

        # source data
        dirs['sousig_train'] = dirs['data'] + 'LibriSpeech/train-clean-100/LibriSpeech/train-clean-100'
        dirs['sousig_test'] = dirs['data'] + 'LibriSpeech/test-clean/LibriSpeech/test-clean'
        dirs['sousig_dev'] = dirs['data'] + 'LibriSpeech/dev-clean/LibriSpeech/dev-clean'

        # noise data
        dirs['noisig_train'] = dirs['data'] + 'Noises-master/NoiseX-92'
        dirs['noisig_test'] = dirs['data'] + 'Noises-master/NoiseX-92'
        dirs['noisig_dev'] = dirs['data'] + 'Noises-master/NoiseX-92'

        # generated data
        dirs['sensig_train'] = dirs['data'] + 'generated_data/train'
        dirs['sensig_dev'] = dirs['data'] + 'generated_data/dev'
        dirs['sensig_test'] = dirs['data'] + 'generated_data/test'

        dirs['sensig_locata'] = dirs['data'] + 'LOCATA'

        return dirs

if __name__ == '__main__':
    opts = opt()
    args = opts().parse()
    dirs = opts().dir()
    print('gpu-id: ' + str(args.gpu_id))
    print('code path:' + dirs['code'])