import numpy as np
from pydub import AudioSegment

def remove_dc_offset(audio_segment):
    samples = np.array(audio_segment.get_array_of_samples())
    samples = samples - round(np.mean(samples))
    return(AudioSegment(samples.tobytes(), channels=audio_segment.channels, sample_width=audio_segment.sample_width, frame_rate=audio_segment.frame_rate))
