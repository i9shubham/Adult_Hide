import glob
import os

import numpy as np
import splitfolders
import tensorflow.keras.models
from PIL import Image, ImageFile
# from keras.preprocessing.image import ImageDataGenerator
import keras
from tensorflow.keras.layers import Flatten, Dense, Conv2D, MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
import tensorflow_hub as hub
import extract_features
import video_processing_operations
import youtube_downloader
from nsfw_detector import predict

ImageFile.LOAD_TRUNCATED_IMAGES = True


# TARGET_SIZE = (224, 224)
# model1 = create_model((224, 224, 3))
# train_generator, valid_generator, test_generator = prepare_model(TARGET_SIZE)

def create_model(input_shape):
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(2, activation='softmax'))
    model.compile(loss='binary_crossentropy', optimizer=Adam(10e-5), metrics=['accuracy'])
    return model


def preprocess_image(image_path, target_size):
    with Image.open(image_path) as img:
        resized_img = img.resize(target_size)
        img_array = np.array(resized_img)
        normalized_img_array = img_array / 255.0
    return normalized_img_array


def prepare_model(target_size, train_dir, valid_dir, test_dir):
    train_datagen = ImageDataGenerator(rescale=1. / 255)
    valid_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    train_generator = train_datagen.flow_from_directory(
        directory=train_dir,
        target_size=target_size,
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    valid_generator = valid_datagen.flow_from_directory(
        directory=valid_dir,
        target_size=target_size,
        color_mode="rgb",
        batch_size=32,
        class_mode="categorical",
        shuffle=True,
        seed=42
    )
    test_generator = test_datagen.flow_from_directory(
        directory=test_dir,
        target_size=target_size,
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42
    )
    return train_generator, valid_generator, test_generator


def prepare_input(path, target_size):
    input_datagen = ImageDataGenerator(1. / 255)
    input_generator = input_datagen.flow_from_directory(
        directory=path,
        target_size=target_size,
        color_mode="rgb",
        batch_size=1,
        class_mode=None,
        shuffle=False,
        seed=42
    )
    return input_generator


def split_folders(ratio):
    splitfolders.ratio('./dataset', output="split_dataset", seed=42, ratio=ratio, group_prefix=None, move=False)


def train_model(model, train_generator, valid_generator):
    step_size_train = train_generator.n
    step_size_valid = valid_generator.n
    model.fit_generator(generator=train_generator,
                        steps_per_epoch=step_size_train // 32,
                        validation_data=valid_generator,
                        validation_steps=step_size_valid // 32,
                        epochs=10
                        )
    model.evaluate_generator(generator=valid_generator,
                             steps=step_size_valid // 32)
    model.save('model.keras')
    model.save_weights('model.h5')
    return model


def test_model(model, test_generator):
    step_size_test = test_generator.n // test_generator.batch_size
    test_generator.reset()
    pred = model.predict_generator(test_generator,
                                   steps=step_size_test,
                                   verbose=1)
    return pred


def match_filename_with_prediction(filepath, prediction):
    return dict(zip(extract_features.get_file_names(filepath), prediction))


def clean():
    frames = glob.glob('./inputs/frames/1/*.jpg')
    video = glob.glob('./inputs/video/*.mp4')
    frames1 = glob.glob('./output/frames/*.jpg')
    video1 = glob.glob('./output/video/*.mp4')
    if len(frames) > 0:
        for f in frames:
            os.chmod(f, 0o777)
            os.remove(f)
    if len(video) > 0:
        os.remove(video[0])
    if len(frames1) > 0:
        for f in frames1:
            os.chmod(f, 0o777)
            os.remove(f)
    if len(video1) > 0:
        os.remove(video1[0])


def load_images(image_paths, image_size, verbose=True):
    loaded_images = []
    loaded_image_paths = []
    if os.path.isdir(image_paths):
        parent = os.path.abspath(image_paths)
        image_paths = [os.path.join(parent, f) for f in os.listdir(image_paths) if
                       os.path.isfile(os.path.join(parent, f))]
    elif os.path.isfile(image_paths):
        image_paths = [image_paths]
    for img_path in image_paths:
        try:
            if verbose:
                print(os.path.basename(img_path), "size:", image_size)
            image = keras.preprocessing.image.load_img(img_path, target_size=image_size)
            image = keras.preprocessing.image.img_to_array(image)
            image /= 255
            loaded_images.append(image)
            loaded_image_paths.append(img_path)
        except Exception as ex:
            print("image load failure: ", img_path, ex)
    return np.asarray(loaded_images), loaded_image_paths


def classify(model, input_paths, image_dim=224):
    images, image_paths = load_images(input_paths, (image_dim, image_dim))
    probs = classify_nd(model, images)
    image_paths = [os.path.basename(image_path) for image_path in image_paths]
    return dict(zip(image_paths, probs))


def classify_nd(model, nd_images):
    model_preds = model.predict(nd_images)
    # 1 and 3 are safe, others unsafe
    categories = ['1', '2', '3', '4', '5']
    probs = []
    for i, single_preds in enumerate(model_preds):
        single_probs = {}
        for j, pred in enumerate(single_preds):
            single_probs[categories[j]] = float(pred)
        probs.append(single_probs)
    return probs


def predict_per_frame():
    video = glob.glob('./inputs/video/*.mp4')
    extract_features.extract_frames(video[0], './inputs/frames/1', 1)
    # input_generator = prepare_input('./inputs/frames', (224, 224))
    model = tensorflow.keras.models.load_model('D://BE Project/adult_content_detection/adult_content_detection/src/saved_model.h5', custom_objects={'KerasLayer': hub.KerasLayer})
    # pred = test_model(model, input_generator)
    pred = classify(model, './inputs/frames/1')
    # safe_unsafe_pred = match_filename_with_prediction('./inputs/frames/1', pred)
    return pred


def ytvideo_predict_per_frame(url):
    clean()
    youtube_downloader.download_video(url, './inputs/video')
    return predict_per_frame()


def video_to_output(out):
    video_processing_operations.blur_frames('./inputs/frames/1/', 'D://BE Project/adult_content_detection/adult_content_detection/src/output/frames/', out)
    video_processing_operations.frames_to_video('D://BE Project/adult_content_detection/adult_content_detection/src/output/frames/', 30, 'D://BE Project/adult_content_detection/adult_content_detection/src/output/video/')
    return '../output/video/video.mp4'


# some links to test with
# https://www.youtube.com/watch?v=eAR2V7PZiIQ
# https://www.youtube.com/watch?v=pZs4SYfU6pA
# https://www.youtube.com/watch?v=bXlQ3Mw4uGc(Short safe video)
def main():
    pass
    # clean()
    # out = ytvideo_predict_per_frame("https://www.youtube.com/watch?v=lSOnPsHd4_M")
    # video_to_output(out)
    # print(out)


if __name__ == '__main__':
    main()
