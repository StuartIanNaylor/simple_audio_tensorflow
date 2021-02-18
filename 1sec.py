import sox
import os

mypath = "/home/stuart/simple_audio_tensorflow/data/speech_commands_v0.02/heymarvin/" 

from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for file in onlyfiles:
  duration = sox.file_info.duration(mypath + file)
  if duration != 1:
    os.remove(mypath + file)
  
