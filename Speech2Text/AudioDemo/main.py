import pyaudio
import wave
import numpy as np
from queue import Queue
import copy
from eval import translate
import threading

class TranslatingThread(threading.Thread):
    
    def __init__(self, data, rate, model_name):
        super().__init__()
        self.data = data
        self.rate = rate
        self.model_name = model_name
    
    def run(self):
        translation = translate(self.data, self.rate, self.model_name)
        print(translation)

def IsActivate(activation_buffer, rate, chunk, last_sec, lb):
    buffer = list(activation_buffer.queue)
    volume = [np.max(np.abs(buf)) for buf in buffer][-int(rate/chunk*last_sec):]
    if np.max(volume) < lb:
        return False
    else:
        return True

def SaveBuffer(buffer, filename, channels, audio_format, rate):
    frames = buffer.queue

    waveFile = wave.open(filename, 'wb')
    waveFile.setnchannels(channels)
    waveFile.setsampwidth(audio.get_sample_size(audio_format))
    waveFile.setframerate(rate)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()

def BufferOutput(buffer):
    frames = buffer.queue
    output = []
    for frame in frames:
        output += np.frombuffer(frame, dtype=np.int16).tolist()
    return output
   
if __name__ == "__main__":

    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    ACTIVATION_BUFFER = 5
    WAVE_OUTPUT_FILENAME = "./temp.wav"
    PRETRAINED_MODEL_NAME = "facebook/s2t-small-librispeech-asr"
    
    audio = pyaudio.PyAudio()
    
    # start Recording
    stream = audio.open(format=FORMAT, channels=CHANNELS,
                    rate=RATE, input=True,
                    frames_per_buffer=CHUNK)
    
    isMonitoring = True
    isActivate = False
    prevActivate = False
    
    activation_buffer = Queue(maxsize=RATE/CHUNK*ACTIVATION_BUFFER)
    buffer = Queue()
    #print(RATE/CHUNK*ACTIVATION_BUFFER)
    
    print("monitoring..")
    while(isMonitoring):
        data = stream.read(CHUNK)
        
        numpydata = np.frombuffer(data, dtype=np.int16)
        if activation_buffer.full():
            activation_buffer.get()
        activation_buffer.put(numpydata)
        
        isActivate = IsActivate(activation_buffer, RATE, CHUNK, 2, lb=15000)
        
        if isActivate:
            print("recording...", end="\r")
            buffer.put(data)
        
        if(prevActivate and not isActivate):
            #SaveBuffer(buffer, WAVE_OUTPUT_FILENAME, CHANNELS, FORMAT, RATE)
            inputs = BufferOutput(buffer)            
            #inputs = (np.array(inputs) / 32767) * np.max(np.abs(inputs))
            inputs = (inputs - np.mean(inputs)) / np.std(inputs)
            t = TranslatingThread(inputs, RATE, PRETRAINED_MODEL_NAME)
            t.start()
            buffer.queue.clear()
            print("sentence end")
        
        prevActivate = isActivate
        
        #print(f"volume: {np.max(np.abs(numpydata))}")
        #print(f"queue size: {activation_buffer.qsize()}")
    