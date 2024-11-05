import os
import json
import librosa

def build_manifest(subset_paths, subset):
    for subset_path in subset_paths:
        for root, dirs, files in os.walk(subset_path):
            for file in files:
                if file.endswith('.txt'):
                    transcript_path = os.path.join(root, file)
                    with open(transcript_path, 'r') as fin:
                        with open('data/' + subset + '-manifest.json', 'a') as fout:
                            for line in fin.readlines():
                                audio_id, transcript = line.split(' ', 1)
                                audio_filepath = os.path.join(root, audio_id + '.flac')
                                duration = librosa.core.get_duration(filename=audio_filepath)


                                metadata = {
                                    "audio_filepath": audio_filepath,
                                    "text": transcript.strip(),
                                    "duration": duration,
                                }
                                json.dump(metadata, fout)
                                fout.write('\n')

def main():
    build_manifest(['/home/hojo/espnet_ctc/egs2/librispeech_100/asr1/downloads/LibriSpeech/train-clean-100'], 'train-clean-100')
    build_manifest([
        '/home/hojo/espnet_ctc/egs2/librispeech_100/asr1/downloads/LibriSpeech/dev-clean',
        '/home/hojo/espnet_ctc/egs2/librispeech_100/asr1/downloads/LibriSpeech/dev-other'
        ], 'dev')
    build_manifest(['/home/hojo/espnet_ctc/egs2/librispeech_100/asr1/downloads/LibriSpeech/test-clean'], 'test-clean')
    build_manifest(['/home/hojo/espnet_ctc/egs2/librispeech_100/asr1/downloads/LibriSpeech/test-other'], 'test-other')

if __name__ == '__main__':
    main()

