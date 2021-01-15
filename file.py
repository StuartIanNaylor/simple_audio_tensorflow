mypath = "data/mini_speech_commands/up/"
from os import listdir
from os.path import isfile, join
onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
for file in onlyfiles:
  sample_ds = preprocess_dataset([mypath + str(file)])

  for spectrogram, label in sample_ds.batch(1):
    prediction = model(spectrogram)
    confidence = tf.nn.softmax(prediction[0])
    if confidence[0] < .4:
      os.remove(mypath + str(file))
