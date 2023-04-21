import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.config import load_config
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models import setup_model

def main():
    torch.set_num_threads(24)

    # pylint: disable=W0105
    """
        This recipe replicates the first experiment proposed in the YourTTS paper (https://arxiv.org/abs/2112.02418).
        YourTTS model is based on the VITS model however it uses external speaker embeddings extracted from a pre-trained speaker encoder and has small architecture changes.
        In addition, YourTTS can be trained in multilingual data, however, this recipe replicates the single language training using the VCTK dataset.
        If you are interested in multilingual training, we have commented on parameters on the VitsArgs class instance that should be enabled for multilingual training.
        In addition, you will need to add the extra datasets following the VCTK as an example.
    """
    CURRENT_PATH = os.path.dirname(os.path.abspath(__file__))

    # Name of the run for the Trainer
    RUN_NAME = "YourTTS-EN-Worgen"

    # If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
    RESTORE_PATH = r"C:\Users\iambl\AppData\Local\tts\tts_models--multilingual--multi-dataset--your_tts\model_file.pth"
    config_file = os.path.join(CURRENT_PATH, "config.json")

    # Or, to continue previous run
    #RESTORE_PATH = os.path.join(CURRENT_PATH, "checkpoints", "vits_worgen-March-25-2023_04+38PM-5293f924", "best_model.pth")
    #config_file = os.path.join(os.path.dirname(RESTORE_PATH), "config.json")

    # This paramter is usefull to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
    SKIP_TRAIN_EPOCH = False

    # Set here the batch size to be used in training and evaluation
    #BATCH_SIZE = 32
    BATCH_SIZE = 8 # My poor 8GB GPU can't handle a batch size of 32!
    
    # init configs
    config = load_config(config_file)
    config.batch_size = BATCH_SIZE
    config.eval_batch_size = BATCH_SIZE
    
    train_args = TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH)
    
    # load data
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    
    # Load model
    model = setup_model(config, train_samples + eval_samples)

    # Init the trainer and 🚀
    trainer = Trainer(
        train_args,
        model.config,
        config.output_path,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
        parse_command_line_args=False,
    )
    trainer.fit()

#os.environ["CUDA_VISIBLE_DEVICES"]="1"
#os.environ["CUDA_LAUNCH_BLOCKING"]="1"

# Some memory management options to help with my poor 8GB GPU
# https://pytorch.org/docs/stable/notes/cuda.html#memory-management
memory_options = {
    'max_split_size_mb': 32,
    'roundup_power2_divisions': 4,
    'roundup_bypass_threshold_mb': 64,
    'garbage_collection_threshold': 0.8,
}
os.environ["PYTORCH_CUDA_ALLOC_CONF"]=','.join(f'{k}:{v}' for k,v in memory_options.items())

from multiprocessing import Process, freeze_support
if __name__ == '__main__':
    freeze_support()  # needed for Windows
    main()