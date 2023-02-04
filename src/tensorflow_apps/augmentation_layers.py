import tensorflow as tf

image_size = 256
crop_size = image_size / 2

augmentation = tf.keras.Sequential([
    tf.keras.layers.Resizing(image_size, image_size),
    tf.keras.layers.RandomRotation(0.2),
    tf.keras.layers.Rescaling(1. / 255),
    tf.keras.layers.RandomZoom(0.5),
    tf.keras.layers.RandomFlip(),
    tf.keras.layers.RandomCrop(crop_size, crop_size),
])
