import os
import numpy as np
from keras.models import load_model
from keras.utils import load_img, img_to_array

model_path = 'CNN_aug_best_weights.h5'
loaded_model = load_model(model_path)

test_image = "C:\\Users\\hp\\Desktop\\weatherdataset\\tests\\alien_test"
test_preprocessed_images = []

for filename in os.listdir(test_image):
    img_path = os.path.join(test_image, filename)
    img = load_img(img_path, target_size=(256, 256))
    img_array = img_to_array(img)
    test_preprocessed_images.append(img_array)

test_preprocessed_images = np.array(test_preprocessed_images)

predictions = loaded_model.predict(test_preprocessed_images)
true_preds = []

for i, pred_ in enumerate(predictions):
    print(f"Images {i + 1} preds ")

    for j, prob in enumerate(pred_):
        print(f"Class {j}: Possibility = {prob:.4f}")
    print("\n")

predicted_class = np.argmax(predictions)

class_labels = ["Class0", "Class1", "Class2", "Class3", "Class4"]

print(f"Most predicted class: {class_labels[predicted_class]}")
