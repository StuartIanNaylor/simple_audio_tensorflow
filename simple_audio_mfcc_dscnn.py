import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import tensorflow as tf

DATASET_FILE = 'data/0-9up.32с.npz' 


dset = np.load(DATASET_FILE)
print(dset['x_train'].shape)

x_train, x_test, x_valid = (
    dset[i].reshape(-1, 49, 13)[:,1:-1]
    for i in ['x_train', 'x_test', 'x_valid']
)
y_train = dset['y_train']
y_test  = dset['y_test']
y_valid = dset['y_valid']

def spectrogram_masking(spectrogram, dim=1, masks_number=2, mask_max_size=5):
  """Spectrogram masking on frequency or time dimension.
  Args:
    spectrogram: Input spectrum [batch, time, frequency]
    dim: dimension on which masking will be applied: 1 - time; 2 - frequency
    masks_number: number of masks
    mask_max_size: mask max size
  Returns:
    masked spectrogram
  """
  if dim not in (1, 2):
    raise ValueError('Wrong dim value: %d' % dim)
  input_shape = spectrogram.shape
  time_size, frequency_size = input_shape[1:3]
  dim_size = input_shape[dim]  # size of dimension on which mask is applied
  stripe_shape = [1, time_size, frequency_size]
  for _ in range(masks_number):
    mask_end = tf.random.uniform([], 0, mask_max_size, tf.int32)
    mask_start = tf.random.uniform([], 0, dim_size - mask_end, tf.int32)

    # initialize stripes with stripe_shape
    stripe_ones_left = list(stripe_shape)
    stripe_zeros_center = list(stripe_shape)
    stripe_ones_right = list(stripe_shape)

    # update stripes dim
    stripe_ones_left[dim] = dim_size - mask_start - mask_end
    stripe_zeros_center[dim] = mask_end
    stripe_ones_right[dim] = mask_start

    # generate mask
    mask = tf.concat((
        tf.ones(stripe_ones_left, spectrogram.dtype),
        tf.zeros(stripe_zeros_center, spectrogram.dtype),
        tf.ones(stripe_ones_right, spectrogram.dtype),
    ), dim)
    spectrogram = spectrogram * mask
  return spectrogram
  
plt.rc('figure', figsize=(13, 4))
  
plt.imshow(x_train[0].T)
  
plt.imshow(spectrogram_masking(x_train[0:1], 1, 3, 3).numpy()[0].T)
  
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.shuffle(len(x_train))
train_dataset = train_dataset.batch(512)
train_dataset = train_dataset.map(lambda x, y: (spectrogram_masking(x, 1, 3, 3), y))
train_dataset = train_dataset.map(lambda x, y: (spectrogram_masking(x, 2, 2, 2), y))

x = x_in = keras.Input(shape=(47, 13))

x = keras.layers.Reshape((47, 1, 13))(x)

x = keras.layers.Conv2D(64, 1, padding="same", use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)
x = keras.layers.SpatialDropout2D(0.1)(x)

for p, f in zip([2, 2, 12], [64, 64, 64]):
  x = keras.layers.DepthwiseConv2D([3, 1], padding="same", use_bias=False)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.SeparableConv2D(f, [3, 1], padding="same", use_bias=False)(x)
  x = keras.layers.BatchNormalization()(x)
  x = keras.layers.ReLU()(x)
  x = keras.layers.SpatialDropout2D(f / 640)(x)
  x = keras.layers.AveragePooling2D([p, 1], padding="same")(x)

x = keras.layers.Flatten()(x)

x = keras.layers.Dense(32, use_bias=False)(x)
x = keras.layers.BatchNormalization()(x)
x = keras.layers.ReLU()(x)

x = keras.layers.Dense(12, activation='softmax')(x)

model = keras.Model(inputs=x_in, outputs=x)

model.summary()

model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])
              
early_stopping = keras.callbacks.EarlyStopping(
        monitor='val_loss',
        mode='min',
        verbose=1,
        patience=20,
        restore_best_weights=True)

history = model.fit(train_dataset,
                    validation_data=(x_valid, y_valid),
                    callbacks=[early_stopping],
                    verbose=2,
                    epochs=1000)
                    
print(history.history.keys())

# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='lower right')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Valid'], loc='upper right')
plt.show()

#dict_keys(['loss', 'accuracy', 'val_loss', 'val_accuracy'])

results = model.evaluate(x_train, y_train, verbose=0)
print('train loss, train acc:', results)

results = model.evaluate(x_test, y_test, verbose=0)
print('test loss, test acc:', results)

results = model.evaluate(x_valid, y_valid, verbose=0)
print('valid loss, valid acc:', results)

assert(len(x_test) + len(x_valid) == 7141)
pred = model.predict(x_test).argmax(axis=-1)
print(np.sum(pred != y_test), len(pred))
pred = model.predict(x_valid).argmax(axis=-1)
print(np.sum(pred != y_valid), len(pred))

model = keras.Sequential([keras.Input(model.layers[2].input_shape[1:])] + model.layers[2:])
model.summary()

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open("dcnn.tflite", "wb") as f:
  f.write(tflite_model)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
def representative_data_gen():
  for x, y in train_dataset.unbatch().batch(1).take(100):
    yield [tf.reshape(x, (-1,) + model.input_shape[1:])]
converter.representative_dataset = representative_data_gen
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
tflite_quant_model = converter.convert()
with open("dcnn.quant.tflite", "wb") as f:
  f.write(tflite_quant_model)
  
def test_micro_model(model):

  with open("valid_test_x.txt", "w") as txt:
    for i in np.concatenate((x_valid, x_test), axis=0):
      txt.write(' '.join(i.flatten().astype(np.str)) + '\n')

  #!./guess "$model" < valid_test_x.txt > valid_test_y.txt
  pred = np.loadtxt("valid_test_y.txt").argmax(axis=-1)
  assert(len(pred) == 7141)

  print(np.sum(pred[len(y_valid):] != y_test), len(y_test))
  print(np.sum(pred[:len(y_valid)] != y_valid), len(y_valid), end = '\n\n')

  true = np.concatenate((y_valid, y_test), axis=0).astype(np.int)
  K = len(np.unique(true))
  assert(K == 12)
  matrix = np.zeros((K, K), dtype=np.int)

  for i in range(len(true)):
    matrix[true[i]][pred[i]] += 1

  for r in matrix:
    l = np.sum(r)
    for i in r:
      print('%.2f' % (i / l), end = ' ')
    print("|", l)

  print()

test_micro_model("dcnn.tflite")

test_micro_model("dcnn.quant.tflite")
 


