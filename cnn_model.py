import numpy as np
import os
import tensorflow as tf
# ... (lanjutan kode lainnya)import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dense, Dropout, Flatten
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Parameter Training
INIT_LR = 1e-4
EPOCHS = 5
BS = 16
IMAGE_SIZE = (224, 224)


def train_model(dataset_dir, model_save_path):
  print(f'\n--- Melatih Model untuk: {dataset_dir} ---')

  if not os.path.exists(dataset_dir):
    print(f'[ERROR] Folder {dataset_dir} tidak ditemukan!')
    return

  train_datagen = ImageDataGenerator(
      rescale=1.0 / 255,
      rotation_range=20,
      zoom_range=0.15,
      width_shift_range=0.2,
      height_shift_range=0.2,
      horizontal_flip=True,
      validation_split=0.2,
      fill_mode='nearest',
  )

  train_generator = train_datagen.flow_from_directory(
      dataset_dir,
      target_size=IMAGE_SIZE,
      batch_size=BS,
      class_mode='binary',
      subset='training',
  )

  val_generator = train_datagen.flow_from_directory(
      dataset_dir,
      target_size=IMAGE_SIZE,
      batch_size=BS,
      class_mode='binary',
      subset='validation',
  )

  baseModel = MobileNetV2(
      weights='imagenet',
      include_top=False,
      input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
  )

  headModel = baseModel.output
  headModel = AveragePooling2D(pool_size=(7, 7))(headModel)
  headModel = Flatten(name='flatten')(headModel)
  headModel = Dense(128, activation='relu')(headModel)
  headModel = Dropout(0.5)(headModel)
  headModel = Dense(1, activation='sigmoid')(headModel)

  model = Model(inputs=baseModel.input, outputs=headModel)

  for layer in baseModel.layers:
    layer.trainable = False

  opt = tf.keras.optimizers.Adam(learning_rate=INIT_LR)
  model.compile(
      loss='binary_crossentropy', optimizer=opt, metrics=['accuracy']
  )

  print('[INFO] Memulai proses training...')
  model.fit(
      train_generator,
      steps_per_epoch=max(1, train_generator.samples // BS),
      validation_data=val_generator,
      validation_steps=max(1, val_generator.samples // BS),
      epochs=EPOCHS,
  )

  os.makedirs(os.path.dirname(model_save_path), exist_ok=True)
  model.save(model_save_path)
  print(f'[INFO] Model berhasil disimpan di: {model_save_path}\n')


if __name__ == '__main__':
  # Buat folder models di luar atau sesuaikan jalurnya
  os.makedirs('../models', exist_ok=True)

  # Latih model mata dan insang (karena script di dalam folder dataset, langsung panggil nama foldernya)
  train_model('mata', '../models/eye_model.h5')
  train_model('insang', '../models/gill_model.h5')
  import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image


def prediksi_citra_asli(model, image_path):
  # Fungsi untuk memprediksi satu gambar uji
  img = image.load_img(image_path, target_size=(224, 224))
  x = image.img_to_array(img)
  x = np.expand_dims(x, axis=0)
  x = x / 255.0

  prediction = model.predict(x)[0][0]
  return float(prediction)
from tensorflow.keras.models import load_model


def build_cnn_model(model_path='../models/eye_model.h5'):
  # Memuat model (.h5) yang sudah dilatih dengan nilai default
  return load_model(model_path)