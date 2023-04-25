from TTS.utils.synthesizer import Synthesizer
from TTS.config import load_config

import pyaudio
import threading
import time

import argparse
import os
import keyboard

import numpy as np

def read_args():
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add the command line arguments
    parser.add_argument('--model_path', type=str, default=r"C:\Users\iambl\AppData\Local\tts\tts_models--multilingual--multi-dataset--your_tts\model_file.pth",
                        help='Path to model file')
    parser.add_argument('--speaker_embeddings_file', type=str, default=r"C:\Users\iambl\GitHub\TTS\recipes\vctk\yourtts\VCTK\speakers.pth",
                        help='Path to speaker embeddings file')

    # Parse the arguments
    args = parser.parse_args()

    # Check if model_base_path is a valid directory
    if not os.path.isfile(args.model_path):
        print(f'{args.model_path} is not a valid directory')
        exit()

    # Check if speaker_embeddings_file is a valid file
    if not os.path.isfile(args.speaker_embeddings_file):
        print(f'{args.speaker_embeddings_file} is not a valid file')
        exit()

    # Print the arguments
    print(f'Model path: {args.model_path}')
    print(f'Speaker embeddings file: {args.speaker_embeddings_file}')

    model_base_path = os.path.dirname(args.model_path)
    return { 
        'model_base_path': model_base_path,
        'model_path': args.model_path, 
        'config_path': os.path.join(model_base_path, 'config.json'),
        'speakers_file_path': os.path.join(model_base_path, 'speakers.json'), 
        'language_ids_file_path': os.path.join(model_base_path, 'language_ids.json'),
        'speaker_embeddings_file': args.speaker_embeddings_file
    }

def prompt_speaker(speakers):
    print("Found the following speakers:")

    # Print the options with right-justified numbers
    for i, speaker in enumerate(speakers):
        print(f"{i+1:>{len(str(len(speakers)))}}. {speaker}")

    # Prompt the user to enter an option
    while True:
        user_input = input("Enter a speaker number or name: ")
        if user_input.isdigit():  # User entered a number
            speaker_index = int(user_input) - 1  # Convert to zero-based index
            if 0 <= speaker_index < len(speakers):
                selected_speaker = speakers[speaker_index]
                break
        else:  # User entered a name
            if user_input in speakers:
                selected_speaker = user_input
                break
        print("Invalid speaker option, please try again.")

    print(f"You selected: {selected_speaker}")

    return selected_speaker

def get_synthesizer(config):
    os.chdir(config['model_base_path'])
    tts_config = load_config(config['config_path'])
    tts_config.d_vector_file = config['speaker_embeddings_file']
    tts_config.model_args.d_vector_file = config['speaker_embeddings_file']
    tts_config.save_json('config_tmp.json')
    synthesizer = Synthesizer(
        config['model_path'],
        'config_tmp.json',
        config['speakers_file_path'],
        config['language_ids_file_path'],
        use_cuda=True,
    )
    os.remove('config_tmp.json')
    return synthesizer

# Define a named function for the audio playback callback
def audio_playback_callback(in_data, frame_count, time_info, status):
    global audio_data
    byte_count = frame_count * p.get_sample_size(pyaudio.paFloat32)
    data = b''
    if audio_data:
        data = audio_data[:byte_count]
        audio_data = audio_data[byte_count:]
    return (data, pyaudio.paContinue)

# Define a named function for the audio recording callback
def audio_recording_callback(in_data, frame_count, time_info, status):
    global audio_data
    audio_data += in_data
    return (None, pyaudio.paContinue)

# Initialize PyAudio
p = pyaudio.PyAudio()

# Define a function to construct the audio stream
def play_audio(sample_rate):
    global audio_stream, byte_rate, is_recording
    is_recording = False
    byte_rate = sample_rate * p.get_sample_size(pyaudio.paFloat32)
    audio_stream = p.open(format=pyaudio.paFloat32,
                    channels=1,
                    rate=sample_rate,
                    output=True,
                    stream_callback=audio_playback_callback)
    resume_audio()
    
# Define a function to pause audio playback
def pause_audio():
    global audio_stream
    if audio_stream and audio_stream.is_active():
        audio_stream.stop_stream()
        print("Paused audio playback")
        
# Define a function to resume audio playback
def resume_audio():
    global audio_stream, byte_rate
    if audio_stream:
        audio_stream.start_stream()
        print(f"Playing audio of length: {len(audio_data) / byte_rate:.2f}s")
    
# Define a function to stop audio playback
def stop_audio():
    global audio_stream, audio_data
    if audio_stream:
        audio_stream.stop_stream()
        audio_stream.close()
    audio_stream = None
    if is_recording:
        print(f"Recorded audio of length: {len(audio_data) / byte_rate:.2f}s")
    else:
        audio_data = b''

# Define a function to construct the audio stream
def record_audio(sample_rate):
    global audio_stream, audio_data, byte_rate, is_recording
    is_recording = True
    audio_data = b''
    byte_rate = sample_rate * p.get_sample_size(pyaudio.paInt16)
    audio_stream = p.open(format=pyaudio.paInt16,
                    channels=1,
                    rate=sample_rate,
                    input=True,
                    stream_callback=audio_recording_callback)

# Checks if there's data in the audio pipeline, either paused or playing
def is_active():
    global audio_stream, audio_data
    return audio_stream and audio_data

# Checks if there's currently audio playing
def is_playing():
    global audio_stream
    return audio_stream and audio_stream.is_active() and not audio_stream.is_stopped()

def serialize_audio(float_data):
    data = np.array(float_data, dtype=np.float32).tobytes()
    return data

def deserialize_audio(byte_data):
    data = np.frombuffer(byte_data, dtype=np.int16)
    return data

# Define a function to synthesize text to audio
def synthesize_text(text, synthesizer, speaker):
    global audio_data
    start_time = time.time()
    float_data = synthesizer.tts(text, speaker, "en")
    audio_data = serialize_audio(float_data)
    end_time = time.time()
    print(f'Synthesized audio in {end_time - start_time:.2f} seconds.')
    play_audio(synthesizer.tts_model.config.audio.sample_rate)

# Define a function to synthesize recording to audio
def synthesize_recording(synthesizer, speaker):
    global audio_data
    start_time = time.time()
    recording_data = deserialize_audio(audio_data)
    synthesizer.tts_model.ap.save_wav(recording_data, 'recording.wav')
    float_data = synthesizer.tts(reference_wav='recording.wav', speaker_name=speaker, language_name="en")
    os.remove('recording.wav')
    audio_data = serialize_audio(float_data)
    end_time = time.time()
    print(f'Synthesized audio in {end_time - start_time:.2f} seconds.')
    play_audio(synthesizer.tts_model.config.audio.sample_rate)

def wait_for_enter_up():
    while True:
        event = keyboard.read_event(suppress=True)
        if event.event_type == 'up' and event.name == 'enter':
            break

def main():
    global audio_stream

    # Prompt the user and start the main loop
    config = read_args()
    print("Initializing TTS...")
    synthesizer = get_synthesizer(config)
    print("TTS synthesizer initialized.")
    speakers = list(synthesizer.tts_model.speaker_manager.name_to_id.keys())
    speaker = prompt_speaker(speakers)

    # Get permissions for recording audio
    record_audio(synthesizer.tts_model.config.audio.sample_rate)
    stop_audio()
    
    audio_stream = None
    print("Ready to speak your text:")
    try:
        while True:
            user_input = input('> ')
            if is_active():
                if is_playing():
                    pause_audio()
                elif not user_input:
                    resume_audio()
                    continue
            elif user_input:
                # Synthesize from text
                synth_thread = threading.Thread(target=synthesize_text, args=(user_input, synthesizer, speaker))
                synth_thread.start()
            else:
                # Synthesize from audio
                print("Recording...")
                record_audio(synthesizer.tts_model.config.audio.sample_rate)
                wait_for_enter_up()
                stop_audio()
                synth_thread = threading.Thread(target=synthesize_recording, args=(synthesizer, speaker))
                synth_thread.start()
    except KeyboardInterrupt:
        pass
    finally:
        print("Exiting...")
        stop_audio()
        p.terminate()

if __name__ == '__main__':
    main()