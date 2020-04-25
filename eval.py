import os
import numpy as np
import pandas as pd
from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator

batch_size = 1000

model = load_model('./models/baseline_model.h5')
model.summary()

# classes
classes = sorted(os.listdir('./data/train'))
idx2name = {}
for i, x in enumerate(classes):
    idx2name[i] = x

# load test data
# don't forget shuffle=False
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory('data/test',
                                                  target_size=(32, 32),
                                                  batch_size=batch_size,
                                                  shuffle=False)

# inference
predictions = model.predict_generator(test_generator,
                                      steps=test_generator.n // batch_size,
                                      workers=8,
                                      verbose=1)
predict_classes = np.argmax(predictions, axis=1)

classes = sorted(os.listdir('./data/train'))
idx2name = {}
for i, x in enumerate(classes):
    idx2name[i] = x
idx2name

# sort by filename number not string
filenames = test_generator.filenames
file_numbers = [int(f.replace('unknown/', '').replace('.png', '')) for f in filenames]
sorted_idx = np.argsort(file_numbers)
predict_classes = np.array(predict_classes)
sorted_predict_classes = predict_classes[sorted_idx]

# create submission csv
submissions = pd.DataFrame({
    'id': list(range(1,
                     len(predict_classes) + 1)),
    'label': sorted_predict_classes
})
submissions.to_csv('submits/aaa.csv', index=False, header=True)
