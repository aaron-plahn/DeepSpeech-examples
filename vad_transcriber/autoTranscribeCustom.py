import os
import logging
import sys
import argparse
import wavTranscriber
import numpy as np

# Debug helpers
logging.basicConfig(stream=sys.stderr, level=logging.DEBUG)

def parseArgs():
    parser = argparse.ArgumentParser(description='Transcribe long audio files using webRTC VAD or use the streaming interface')
    parser.add_argument('--aggressive', type=int, choices=range(4), required=False,
                        help='Determines how aggressive filtering out non-speech is. (Interger between 0-3)')
    parser.add_argument('--audio', required=False,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--model', required=True,
                        help='Path to directory that contains all model files (output_graph and scorer)')
    parser.add_argument('--stream', required=False, action='store_true',
                        help='To use deepspeech streaming interface')
    return parser.parse_args()

def main(args):
    args = parseArgs()
    
    # Point to a path containing the pre-trained models & resolve ~ if used
    dirName = os.path.expanduser(args.model)
    # Resolve all the paths of model files
    output_graph, scorer = wavTranscriber.resolve_models(dirName)
    # Load output_graph, alpahbet and scorer
    model_retval = wavTranscriber.load_model(output_graph, scorer)

    if args.audio is None:
        raise Exception("Audio is undefined.")
    waveFile = args.audio

    title_names = ['Filename', 'Duration(s)', 'Inference Time(s)', 'Model Load Time(s)', 'Scorer Load Time(s)']
    print("\n%-30s %-20s %-20s %-20s %s" % (title_names[0], title_names[1], title_names[2], title_names[3], title_names[4]))

    inference_time = 0.0

    # Run VAD on the input file

    segments, sample_rate, audio_length = wavTranscriber.vad_segment_generator(waveFile, args.aggressive)
    f = open(waveFile.rstrip(".wav") + ".txt", 'w')
    logging.debug("Writing Transcript to: %s" % waveFile.rstrip(".wav") + ".txt")

    end_time = 0

    def writeLine(audio_segment_length,content,end_time,file):
        logging.debug("Transcript: %s" % content)
        audio_segment_length = len(audio)/sample_rate
        start_time = end_time
        end_time = start_time + audio_segment_length
        f.write(content + "," + str(start_time) + "," + str(end_time))
        f.write('\n')

    for i, segment in enumerate(segments):
        # Run deepspeech on the chunk that just completed VAD
        logging.debug("Processing chunk %002d" % (i,))
        audio = np.frombuffer(segment, dtype=np.int16)
        output = wavTranscriber.stt(model_retval[0], audio, sample_rate,file=f)
        inference_time += output[1]

        writeLine(len(audio)/sample_rate,output[0],end_time,file=f)

    # Summary of the files processed
    f.close()
    

if __name__ == '__main__':
    main(sys.argv[1:])
