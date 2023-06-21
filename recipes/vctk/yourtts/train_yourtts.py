import os

import torch
from trainer import Trainer, TrainerArgs

from TTS.bin.compute_embeddings import compute_embeddings
from TTS.bin.resample import resample_files
from TTS.config.shared_configs import BaseDatasetConfig
from TTS.tts.configs.vits_config import VitsConfig
from TTS.tts.datasets import load_tts_samples
from TTS.tts.models.vits import CharactersConfig, Vits, VitsArgs, VitsAudioConfig
from TTS.utils.downloaders import download_vctk

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
    RUN_NAME = "Worgen-only"

    # Path where you want to save the models outputs (configs, checkpoints and tensorboard logs)
    OUT_PATH = os.path.dirname(os.path.abspath(__file__))  # "/raid/coqui/Checkpoints/original-YourTTS/"

    # If you want to do transfer learning and speedup your training you can set here the path to the original YourTTS model
    RESTORE_PATH = r"C:\Users\iambl\AppData\Local\tts\tts_models--multilingual--multi-dataset--your_tts\model_file.pth"
    CONFIG_SE_PATH = os.path.join(os.path.dirname(RESTORE_PATH), "config_se.json")
    MODEL_SE_PATH = os.path.join(os.path.dirname(RESTORE_PATH), "model_se.pth")

    # Or, to continue previous run
    RESTORE_PATH = r"C:\Users\iambl\GitHub\TTS\recipes\vctk\yourtts\YourTTS-EN-VCTK-April-24-2023_03+23PM-3fa2634a\checkpoint_65000.pth"

    # This paramter is usefull to debug, it skips the training epochs and just do the evaluation  and produce the test sentences
    SKIP_TRAIN_EPOCH = False

    # Set here the batch size to be used in training and evaluation
    #BATCH_SIZE = 32
    BATCH_SIZE = 8 # My poor 12GB GPU can't handle a batch size of 32!

    # Training Sampling rate and the target sampling rate for resampling the downloaded dataset (Note: If you change this you might need to redownload the dataset !!)
    # Note: If you add new datasets, please make sure that the dataset sampling rate and this parameter are matching, otherwise resample your audios
    SAMPLE_RATE = 16000

    # Max audio length in seconds to be used in training (every audio bigger than it will be ignored)
    MAX_AUDIO_LEN_IN_SECONDS = 11

    ### Download VCTK dataset
    VCTK_DOWNLOAD_PATH = os.path.join(CURRENT_PATH, "VCTK")
    # Define the number of threads used during the audio resampling
    NUM_RESAMPLE_THREADS = 10
    # Check if VCTK dataset is not already downloaded, if not download it
    if not os.path.exists(VCTK_DOWNLOAD_PATH):
        print(">>> Downloading VCTK dataset:")
        download_vctk(VCTK_DOWNLOAD_PATH)
        resample_files(VCTK_DOWNLOAD_PATH, SAMPLE_RATE, file_ext="flac", n_jobs=NUM_RESAMPLE_THREADS)

    WORGEN_PATH = os.path.join(CURRENT_PATH, "Worgen")

# init configs
    vctk_config = BaseDatasetConfig(
        formatter="vctk", dataset_name="vctk", meta_file_train="", meta_file_val="", path=VCTK_DOWNLOAD_PATH, language="en"
    )

    worgen_config = BaseDatasetConfig(
        formatter="multi_ljspeech",
        dataset_name="worgen",
        meta_file_train="metadata.txt",
        meta_file_val="",
        path=WORGEN_PATH,
        language="en",
    )

    # Add here all datasets configs, in our case we just want to train with the VCTK dataset then we need to add just VCTK. Note: If you want to added new datasets just added they here and it will automatically compute the speaker embeddings (d-vectors) for this new dataset :)
    DATASETS_CONFIG_LIST = [vctk_config, worgen_config]
    #DATASETS_CONFIG_LIST = [worgen_config]

    ### Extract speaker embeddings
    SPEAKER_ENCODER_CHECKPOINT_PATH = (
        "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/model_se.pth.tar"
    )
    SPEAKER_ENCODER_CONFIG_PATH = "https://github.com/coqui-ai/TTS/releases/download/speaker_encoder_model/config_se.json"

    D_VECTOR_FILES = []  # List of speaker embeddings/d-vectors to be used during the training

    # Iterates all the dataset configs checking if the speakers embeddings are already computated, if not compute it
    for dataset_conf in DATASETS_CONFIG_LIST:
        # Check if the embeddings weren't already computed, if not compute it
        embeddings_file = os.path.join(dataset_conf.path, "speakers.pth")
        if not os.path.isfile(embeddings_file):
            print(f">>> Computing the speaker embeddings for the {dataset_conf.dataset_name} dataset")
            compute_embeddings(
                SPEAKER_ENCODER_CHECKPOINT_PATH,
                SPEAKER_ENCODER_CONFIG_PATH,
                embeddings_file,
                old_spakers_file=None,
                config_dataset_path=None,
                formatter_name=dataset_conf.formatter,
                dataset_name=dataset_conf.dataset_name,
                dataset_path=dataset_conf.path,
                meta_file_train=dataset_conf.meta_file_train,
                meta_file_val=dataset_conf.meta_file_val,
                disable_cuda=False,
                no_eval=False,
            )
        D_VECTOR_FILES.append(embeddings_file)


    # Audio config used in training.
    audio_config = VitsAudioConfig(
        sample_rate=SAMPLE_RATE,
        hop_length=256,
        win_length=1024,
        fft_size=1024,
        mel_fmin=0.0,
        mel_fmax=None,
        num_mels=80,
    )

    # Init VITSArgs setting the arguments that is needed for the YourTTS model
    model_args = VitsArgs(
        d_vector_file=D_VECTOR_FILES,
        use_d_vector_file=True,
        d_vector_dim=512,
        num_layers_text_encoder=10,
        resblock_type_decoder="2",  # On the paper, we accidentally trained the YourTTS using ResNet blocks type 2, if you like you can use the ResNet blocks type 1 like the VITS model
        # Usefull parameters to enable the Speaker Consistency Loss (SCL) discribed in the paper
        # use_speaker_encoder_as_loss=True,
        # speaker_encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH,
        # speaker_encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH,
        # Usefull parameters to the enable multilingual training
        # use_language_embedding=True,
        # embedded_language_dim=4,
        # freeze_DP=True,
        # freeze_PE=True,
        # freeze_flow_decoder=True,
        # Use same args as pretrained model
        num_chars=165,
        out_channels=513,
        spec_segment_size=62,
        hidden_channels=192,
        hidden_channels_ffn_text_encoder=768,
        num_heads_text_encoder=2,
        kernel_size_text_encoder=3,
        dropout_p_text_encoder=0.1,
        dropout_p_duration_predictor=0.5,
        kernel_size_posterior_encoder=5,
        dilation_rate_posterior_encoder=1,
        num_layers_posterior_encoder=16,
        kernel_size_flow=5,
        dilation_rate_flow=1,
        num_layers_flow=4,
        resblock_kernel_sizes_decoder=[
            3,
            7,
            11
        ],
        resblock_dilation_sizes_decoder=[
            [
                1,
                3,
                5
            ],
            [
                1,
                3,
                5
            ],
            [
                1,
                3,
                5
            ]
        ],
        upsample_rates_decoder=[
            8,
            8,
            2,
            2
        ],
        upsample_initial_channel_decoder=512,
        upsample_kernel_sizes_decoder=[
            16,
            16,
            4,
            4
        ],
        periods_multi_period_discriminator=[
            2,
            3,
            5,
            7,
            11
        ],
        use_sdp=True,
        noise_scale=1.0,
        inference_noise_scale=0.3,
        length_scale=1.5,
        noise_scale_dp=0.6,
        inference_noise_scale_dp=0.3,
        max_inference_len=None,
        init_discriminator=True,
        use_spectral_norm_disriminator=False,
        use_speaker_embedding=False,
        num_speakers=1244,
        speakers_file=None,
        speaker_embedding_channels=512,
        detach_dp_input=True,
        use_language_embedding=True,
        embedded_language_dim=4,
        num_languages=3,
        language_ids_file=None,
        use_speaker_encoder_as_loss=True,
        speaker_encoder_config_path=CONFIG_SE_PATH,
        speaker_encoder_model_path=MODEL_SE_PATH,
        condition_dp_on_speaker=True,
        encoder_sample_rate=None,
        interpolate_z=True,
        reinit_DP=False,
        reinit_text_encoder=False
    )

    # General training config, here you can change the batch size and others usefull parameters
    config = VitsConfig(
        output_path=OUT_PATH,
        model_args=model_args,
        run_name=RUN_NAME,
        project_name="YourTTS",
        run_description="""
                - Original VCTK YourTTS fine-tuned on Worgen dataset
            """,
        dashboard_logger="tensorboard",
        logger_uri=None,
        audio=audio_config,
        batch_size=BATCH_SIZE,
        batch_group_size=48,
        eval_batch_size=BATCH_SIZE,
        num_loader_workers=8,
        eval_split_size=0.05,
        eval_split_max_size=256,
        print_step=50,
        plot_step=100,
        log_model_step=1000,
        save_step=5000,
        save_n_checkpoints=3,
        save_checkpoints=True,
        target_loss="loss_1",
        print_eval=False,
        use_phonemes=False,
        phonemizer="espeak",
        phoneme_language="en",
        compute_input_seq_cache=True,
        add_blank=True,
        text_cleaner="multilingual_cleaners",
        characters=CharactersConfig(
            characters_class="TTS.tts.models.vits.VitsCharacters",
            pad="_",
            eos="&",
            bos="*",
            blank=None,
            characters="ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz\u00af\u00b7\u00df\u00e0\u00e1\u00e2\u00e3\u00e4\u00e6\u00e7\u00e8\u00e9\u00ea\u00eb\u00ec\u00ed\u00ee\u00ef\u00f1\u00f2\u00f3\u00f4\u00f5\u00f6\u00f9\u00fa\u00fb\u00fc\u00ff\u0101\u0105\u0107\u0113\u0119\u011b\u012b\u0131\u0142\u0144\u014d\u0151\u0153\u015b\u016b\u0171\u017a\u017c\u01ce\u01d0\u01d2\u01d4\u0430\u0431\u0432\u0433\u0434\u0435\u0436\u0437\u0438\u0439\u043a\u043b\u043c\u043d\u043e\u043f\u0440\u0441\u0442\u0443\u0444\u0445\u0446\u0447\u0448\u0449\u044a\u044b\u044c\u044d\u044e\u044f\u0451\u0454\u0456\u0457\u0491\u2013!'(),-.:;? ",
            punctuations="!'(),-.:;? ",
            phonemes="",
            is_unique=True,
            is_sorted=True,
        ),
        phoneme_cache_path=None,
        precompute_num_workers=12,
        start_by_longest=True,
        datasets=DATASETS_CONFIG_LIST,
        cudnn_benchmark=False,
        max_audio_len=SAMPLE_RATE * MAX_AUDIO_LEN_IN_SECONDS,
        mixed_precision=False,
        test_sentences=[
            [
                "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "VCTK_p277",
                None,
                "en",
            ],
            [
                "Be a voice, not an echo.",
                "VCTK_p239",
                None,
                "en",
            ],
            [
                "I'm sorry Dave. I'm afraid I can't do that.",
                "VCTK_p258",
                None,
                "en",
            ],
            [
                "This cake is great. It's so delicious and moist.",
                "VCTK_p244",
                None,
                "en",
            ],
            [
                "Prior to November 22, 1963.",
                "VCTK_p305",
                None,
                "en",
            ],
            [
                "It took me quite a long time to develop a voice, and now that I have it I'm not going to be silent.",
                "female-worgen",
                None,
                "en"
            ],
            [
                "It's only fun when they run!",
                "female-worgen",
                None,
                "en"
            ],
            [
                "I am kinder than Miss Alina, though her voice is prettier.",
                "female-worgen",
                None,
                "en"
            ],
            [
                "I need to move, to do something, anything, even if just jogging in place!",
                "female-worgen",
                None,
                "en"
            ]
        ],
        # Enable the weighted sampler
        use_weighted_sampler=True,
        # Ensures that all speakers are seen in the training batch equally no matter how many samples each speaker has
        weighted_sampler_attrs={"speaker_name": 1.0},
        # It defines the Speaker Consistency Loss (SCL) α to 9 like the paper
        speaker_encoder_loss_alpha=9.0,
    )

    # Load all the datasets samples and split traning and evaluation sets
    train_samples, eval_samples = load_tts_samples(
        config.datasets,
        eval_split=True,
        eval_split_max_size=config.eval_split_max_size,
        eval_split_size=config.eval_split_size,
    )
    
    # Add fake language sources to keep language embedder the same size
    config.datasets += [
        BaseDatasetConfig(language='pt-br'),
        BaseDatasetConfig(language='fr-fr')
    ]

    # Init the model
    model = Vits.init_from_config(config)

    print(list(model.speaker_manager.embeddings.keys()))

    # Init the trainer and 🚀
    trainer = Trainer(
        TrainerArgs(restore_path=RESTORE_PATH, skip_train_epoch=SKIP_TRAIN_EPOCH),
        config,
        output_path=OUT_PATH,
        model=model,
        train_samples=train_samples,
        eval_samples=eval_samples,
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