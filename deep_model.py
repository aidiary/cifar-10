from comet_ml import Experiment
import os
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPool2D, Dropout, GlobalAveragePooling2D, BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, Callback

experiment = Experiment(project_name="cifar-10",
                        workspace="kmori",
                        auto_param_logging=False,
                        auto_metric_logging=False,
                        disabled=False)

batch_size = 500
epochs = 150

params = {'batch_size': batch_size, 'epochs': epochs}

experiment.log_parameters(params)

os.makedirs('models', exist_ok=True)

train_datagen = ImageDataGenerator(rescale=1. / 255,
                                   rotation_range=20,
                                   horizontal_flip=True,
                                   height_shift_range=0.2,
                                   width_shift_range=0.2,
                                   zoom_range=0.2,
                                   channel_shift_range=0.2)

train_generator = train_datagen.flow_from_directory('./data/train',
                                                    target_size=(32, 32),
                                                    batch_size=batch_size)

valid_datagen = ImageDataGenerator(rescale=1. / 255)
valid_generator = valid_datagen.flow_from_directory('./data/valid',
                                                    target_size=(32, 32),
                                                    batch_size=batch_size)


def create_deep_model():
    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', activation='relu', input_shape=(32, 32, 3)))
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.25))

    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
    model.add(BatchNormalization())
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(Conv2D(512, (3, 3), padding='same', activation='relu'))
    model.add(GlobalAveragePooling2D())

    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1024, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(10, activation='softmax'))

    model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])

    return model


baseline_model = create_deep_model()
baseline_model.summary()

checkpointer = ModelCheckpoint(filepath='models/deep_model.h5', verbose=1, save_best_only=True)


class MetricsLogger(Callback):
    def on_epoch_end(self, epoch, logs={}):
        experiment.log_metric('loss', logs.get('loss'), step=epoch + 1)
        experiment.log_metric('accuracy', logs.get('accuracy'), step=epoch + 1)
        experiment.log_metric('val_loss', logs.get('val_loss'), step=epoch + 1)
        experiment.log_metric('val_accuracy', logs.get('val_accuracy'), step=epoch + 1)


logger = MetricsLogger()

history = baseline_model.fit_generator(
    train_generator,
    steps_per_epoch=train_generator.n // train_generator.batch_size,
    epochs=epochs,
    validation_data=valid_generator,
    validation_steps=valid_generator.n // valid_generator.batch_size,
    verbose=1,
    callbacks=[checkpointer, logger])
