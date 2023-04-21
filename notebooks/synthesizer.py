from TTS.utils.synthesizer import Synthesizer

import pyaudio
import threading
import time

import argparse
import os

def read_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the command line arguments
    parser.add_argument('--model_base_path', type=str, default=r"C:\Users\iambl\GitHub\TTS\recipes\vctk\yourtts\WorgenVCTK",
                        help='Path to model base directory')
    parser.add_argument('--speaker_embeddings_file', type=str, default=r"C:\Users\iambl\GitHub\TTS\recipes\vctk\yourtts\VCTK\speakers.pth",
                        help='Path to speaker embeddings file')

    # Parse the arguments
    args = parser.parse_args()

    # Check if model_base_path is a valid directory
    if not os.path.isdir(args.model_base_path):
        print(f'{args.model_base_path} is not a valid directory')
        exit()

    # Check if speaker_embeddings_file is a valid file
    if not os.path.isfile(args.speaker_embeddings_file):
        print(f'{args.speaker_embeddings_file} is not a valid file')
        exit()

    # Print the arguments
    print(f'Model base path: {args.model_base_path}')
    print(f'Speaker embeddings file: {args.speaker_embeddings_file}')
    
    return { 
        'model_path': os.path.join(args.model_base_path, 'model_file.pth'), 
        'config_path': os.path.join(args.model_base_path, 'config.json'),
        'speakers_file_path': os.path.join(args.model_base_path, 'speakers.json'), 
        'language_ids_file_path': os.path.join(args.model_base_path, 'language_ids.json'),
        'speaker_embeddings_file': args.speaker_embeddings_file
    }

def prompt_speaker(speaker):
    print("Found the following speakers:")

    # Print the options with right-justified numbers
    for i, speaker in enumerate(speakers):
        print(f"{i+1:>{len(str(len(speakers)))}}. {speaker}")

    # Prompt the user to enter an option
    while True:
        user_input = input("Enter a speaker number or name: ")
        if user_input.isdigit():  # User entered a number
            speaker_index = int(user_input) - 1  # Convert to zero-based index
            if 0 <= speaker_index < len(speaker):
                selected_option = speakers[speaker_index]
                break
        else:  # User entered a name
            if user_input in speakers:
                selected_speaker = user_input
                break
        print("Invalid speaker option, please try again.")

    print(f"You selected: {selected_speaker}")

    return selected_speaker

def get_synthesizer(config):
    synthesizer = Synthesizer(
        config['model_path'],
        config['config_path'],
        config['speakers_file_path'],
        config['language_ids_file_path'],
        use_cuda=True,
    )
    synthesizer.tts_model.speaker_manager.load_embeddings_from_file(config['speaker_embeddings_file'])

# Define a named function for the audio callback
def audio_callback(in_data, frame_count, time_info, status):
    global audio_data
    data = audio_data[:frame_count]
    audio_data = audio_data[frame_count:]
    return (data, pyaudio.paContinue)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define a function to play audio asynchronously
def play_audio_async(audio_data, sample_rate):
    global audio_stream
    audio_stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    output=True,
                    stream_callback=audio_callback)
    audio_stream.start_stream()

# Define a function to pause audio playback
def pause_audio():
    global audio_stream
    if audio_stream and audio_stream.is_active():
        audio_stream.stop_stream()
        
# Define a function to resume audio playback
def resume_audio():
    global audio_stream
    if audio_stream and audio_stream.is_active():
        audio_stream.start_stream()
    
# Define a function to stop audio playback
def stop_audio():
    global audio_stream, audio_data
    if audio_stream and audio_stream.is_active():
        audio_stream.stop_stream()
        audio_stream.close()
    audio_stream = None
    audio_data = b''

# Define a function to synthesize text to audio
def synthesize_text(text, synthesizer, speaker):
    global audio_data
    start_time = time.time()
    audio_data = synthesizer.tts(text, speaker, "en")
    end_time = time.time()
    print(f'Synthesized audio in {end_time - start_time:.2f} seconds.')
    play_audio_async(audio_data, synthesizer.sample_rate)

def main():
    # Prompt the user and start the main loop
    config = read_args()
    print("Initializing TTS...")
    synthesizer = get_synthesizer(config)
    print("TTS synthesizer initialized.")
    speakers = list(synthesizer.tts_model.speaker_manager.name_to_id.keys())
    speaker = prompt_speaker(speakers)
    
    print("Ready to speak your text:")
    audio_data = b''
    audio_stream = None
    try:
        while True:
            user_input = input('> ')
            if not audio_stream.is_stopped():
                pause_audio()
            elif user_input is None:
                resume_audio()
                continue
            if user_input:
                synth_thread = threading.Thread(target=synthesize_text, args=(user_input, synthesizer, speaker))
                synth_thread.start()
    except KeyboardInterrupt:
        stop_audio()
        p.terminate()

if __name__ == '__main__':
    main()