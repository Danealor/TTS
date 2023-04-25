import audacity

files=['C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p225\\p225_017_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p229\\p229_018_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p231\\p231_172_mic2.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p233\\p233_029_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p248\\p248_138_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p252\\p252_231_mic2.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p253\\p253_331_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p258\\p258_409_mic2.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p260\\p260_003_mic2.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p262\\p262_370_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p262\\p262_370_mic2.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p271\\p271_023_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p279\\p279_017_mic2.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p279\\p279_266_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p286\\p286_019_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p288\\p288_018_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p299\\p299_221_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p304\\p304_099_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p311\\p311_070_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p311\\p311_080_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p311\\p311_202_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p311\\p311_374_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p315\\p315_349_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_003_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_005_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_029_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_035_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_047_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_058_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_085_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_094_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_095_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_108_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_131_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_171_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_179_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_186_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_208_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_287_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_306_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_315_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_348_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_353_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_365_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_400_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_404_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p323\\p323_405_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p336\\p336_048_mic2.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\p339\\p339_006_mic1.flac',
 'C:\\Users\\iambl\\GitHub\\TTS\\recipes\\vctk\\yourtts\\VCTK\\wav48_silence_trimmed\\s5\\s5_060_mic1.flac']

aud = audacity.AudacityScript()

def clip_file(file):
    aud.do_command(f'Import2: Filename={file}')
    aud.do_command('Limiter: type=HardLimit thresh="-1"')
    aud.do_command(f'Export2: Filename={file}')
    aud.do_command(f'TrackClose')
    
for file in files:
    clip_file(file)