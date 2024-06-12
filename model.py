import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn.metrics import classification_report, confusion_matrix
import itertools
import seaborn as sns
from google.colab import drive
drive.mount('/content/drive')
base_dir = '/content/drive/My Drive/datasetfinal'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'test')
train_datagen = ImageDataGenerator(
      rescale=1./255,
      rotation_range=40,
      width_shift_range=0.2,
      height_shift_range=0.2,
      shear_range=0.2,
      zoom_range=0.2,
      horizontal_flip=True,
      fill_mode='nearest')

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
        validation_dir,
        target_size=(150, 150),
        batch_size=20,
        class_mode='categorical')
base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(150, 150, 3))
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(22, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Unfreeze some layers
for layer in base_model.layers[-30:]:
    layer.trainable = True

model.compile(optimizer=Adam(lr=1e-4), loss='categorical_crossentropy', metrics=['accuracy'])

# Continue training with the same callbacks
history_fine_tune = model.fit(
    train_generator,
    steps_per_epoch=100,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=50,
    callbacks=callbacks,
    verbose=2)
# Evaluate the model
val_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='categorical',
    shuffle=False)

Y_pred = model.predict(val_generator, val_generator.samples // val_generator.batch_size+1)
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
cm = confusion_matrix(val_generator.classes, y_pred)
print(cm)

# Classification report
print('Classification Report')
target_names = list(val_generator.class_indices.keys())
print(classification_report(val_generator.classes, y_pred, target_names=target_names))

# Plotting training/validation accuracy and loss
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'r', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'r', label='Training Loss')
plt.plot(epochs, val_loss, 'b', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

import matplotlib.pyplot as plt

# Fetch a batch of images and labels
images, labels = next(train_generator)

# Determine the number of images to display (for example, 8 images in a 2x4 grid)
num_images_to_display = 16
plt.figure(figsize=(12, 12))

for i in range(num_images_to_display):
    ax = plt.subplot(4, 4, i+1)
    plt.imshow(images[i])
    plt.title('Label: ' + str(np.argmax(labels[i])))
    plt.axis('off')

plt.tight_layout()
plt.show()

import matplotlib.pyplot as plt
import numpy as np

# Function to plot images in a grid with their predictions
def plot_images_with_predictions(images, true_labels, predictions, class_labels, grid_size=(4, 4)):
    fig, axes = plt.subplots(grid_size[0], grid_size[1], figsize=(12, 12))
    axes = axes.flatten()
    for img, true_label, pred_label, ax in zip(images, true_labels, predictions, axes):
        ax.imshow(img)
        ax.axis('off')
        true_class = class_labels[np.argmax(true_label)]
        pred_class = class_labels[np.argmax(pred_label)]
        ax.set_title(f"True: {true_class}\nPred: {pred_class}")
    plt.tight_layout()
    plt.show()

# Fetch a batch of images and labels from the validation generator
images_batch, labels_batch = next(validation_generator)

# Make predictions on the batch
predictions_batch = model.predict(images_batch)

# Convert one-hot encoded labels to integers
true_labels = np.argmax(labels_batch, axis=1)

# Plotting 16 random images with their true and predicted labels
num_images = 16
random_indices = np.random.choice(images_batch.shape[0], num_images, replace=False)
plot_images_with_predictions(images_batch[random_indices], labels_batch[random_indices], predictions_batch[random_indices], list(validation_generator.class_indices.keys()))
