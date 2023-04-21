def multi_ljspeech(root_path, meta_filename=None, ignored_speakers=None):
    """LJSpeech format adapted for multiple speakers"""
    file_ext = "wav"
    items = []
    if meta_filename is None:
        meta_filename = "metadata.txt"
    meta_files = glob(f"{root_path}/*/{meta_filename}", recursive=True)
    for meta_file in meta_files:
        speaker_id, txt_file = os.path.relpath(meta_file, root_path).split(os.sep)
        file_id = txt_file.split(".")[0]
        # ignore speakers
        if isinstance(ignored_speakers, list):
            if speaker_id in ignored_speakers:
                continue
        with open(meta_file, "r", encoding="utf-8") as file_text:
            for line in file_text:
                cols = line.split("|")
                wav_file = os.path.join(root_path, speaker_id, "wavs", cols[0] + ".wav")
                text = cols[2].strip()
                if os.path.exists(wav_file):
                    items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_id, "root_path": root_path})
                else:
                    print(f" [!] wav files don't exist - {wav_file}")
                items.append({"text": text, "audio_file": wav_file, "speaker_name": speaker_id, "root_path": root_path})
    return items