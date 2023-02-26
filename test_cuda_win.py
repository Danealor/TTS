import torch
import os

def main():
    print(torch.cuda.is_available())

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"
from multiprocessing import Process, freeze_support
if __name__ == '__main__':
    freeze_support()  # needed for Windows
    main()
