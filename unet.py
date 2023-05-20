from keras.utils import load_img, array_to_img, img_to_array
from keras import layers, Model, optimizers, metrics
import tensorflow as tf
import os
import numpy as np
import matplotlib.pyplot as plt

TRAINING_DATA_DIRECTORY = 'training_data'
VALIDATION_DATA_DIRECTORY = 'validation_data'

INPUT_SIZE = 256
INPUT_MARGIN_SIZE = 16
INPUT_CONTENT_SIZE = INPUT_SIZE - INPUT_MARGIN_SIZE * 2


# region LOADING DATA

def import_data_sample(directory, index=1):
    sample = load_img(os.path.join(directory, f'{index}.png'), color_mode='rgba')
    mask = load_img(os.path.join(directory, f'{index}_mask.png'), color_mode='rgba')

    sample_array = img_to_array(sample)
    sample_array = sample_array[:, :, :3]
    sample_array = sample_array / 255.

    mask_array = img_to_array(mask)
    mask_array = mask_array[:, :, 3:4]
    mask_array = np.round(mask_array / 255.)

    return sample_array, mask_array


def get_sample_count(directory):
    return int(len([name for name in os.listdir(directory)]) / 2)


def load_data_samples(directory):
    samples = []
    masks = []

    for i in range(1, get_sample_count(directory) + 1):
        sample, mask = import_data_sample(directory, i)
        samples.append(sample)
        masks.append(mask)

    return samples, masks


def load_training_data_samples():
    return load_data_samples(TRAINING_DATA_DIRECTORY)


def load_validation_data_samples():
    return load_data_samples(VALIDATION_DATA_DIRECTORY)


# endregion


# region PADDING BIG IMAGE

# L = [l/a]*a+2*b
def calculate_target_size(l: int):
    a = float(INPUT_CONTENT_SIZE)
    b = INPUT_MARGIN_SIZE

    return int(np.round(np.ceil(l / a) * a + 2 * b))


def calculate_padding_size(l: int):
    padding_both = calculate_target_size(l) - l

    first = int(padding_both / 2)
    second = padding_both - first

    return first, second


def get_image_size(image: np.array):
    height, width, _ = image.shape

    return width, height


def calculate_target_width_height(image: np.array):
    width, height = get_image_size(image)

    return calculate_target_size(width), calculate_target_size(height)


def pad_image(image: np.array, verbose=0):
    width, height = get_image_size(image)

    target_width, target_height = calculate_target_width_height(image)

    width_pad_start, _ = calculate_padding_size(width)
    height_pad_start, _ = calculate_padding_size(height)

    if verbose == 1:
        print(f'width={width}, height={height}')
        print(f'target_width={target_width}, target_height={target_height}')
        print(f'width_pad_start={width_pad_start}, height_pad_start={height_pad_start}')

    image_padded = tf.image.pad_to_bounding_box(image, height_pad_start, width_pad_start, target_height, target_width)

    return image_padded


def pad_data_sample(sample: np.array, mask: np.array):
    return pad_image(sample), pad_image(mask)


# endregion


# region PAD & CROP BIG IMAGE

def get_bounding_boxes_start_point(image: np.array):
    target_width, target_height = calculate_target_width_height(image)

    width_points = np.arange(0, target_width - INPUT_SIZE + 1, INPUT_CONTENT_SIZE)
    height_points = np.arange(0, target_height - INPUT_SIZE + 1, INPUT_CONTENT_SIZE)
    xv, yv = np.meshgrid(width_points, height_points)

    return xv.flatten(), yv.flatten()


def display_bounding_boxes(image: np.array):
    x_start, y_start = get_bounding_boxes_start_point(image)
    x_end, y_end = x_start + INPUT_SIZE, y_start + INPUT_SIZE

    x_inner_start, y_inner_start = x_start + INPUT_MARGIN_SIZE, y_start + INPUT_MARGIN_SIZE
    x_inner_end, y_inner_end = x_inner_start + INPUT_CONTENT_SIZE, y_inner_start + INPUT_CONTENT_SIZE

    target_width, target_height = calculate_target_width_height(image)

    boxes = np.array([(y_min, x_min, y_max, x_max) for x_min, y_min, x_max, y_max in
                      zip(x_start,
                          y_start,
                          x_end,
                          y_end)])
    boxes = np.divide(boxes, [float(target_height), float(target_width), float(target_height), float(target_width)])
    boxes = boxes.reshape([1, len(boxes), 4])

    boxes_inner = np.array([(y_min, x_min, y_max, x_max) for x_min, y_min, x_max, y_max in
                            zip(x_inner_start,
                                y_inner_start,
                                x_inner_end,
                                y_inner_end)])
    boxes_inner = np.divide(boxes_inner,
                            [float(target_height), float(target_width), float(target_height), float(target_width)])
    boxes_inner = boxes_inner.reshape([1, len(boxes_inner), 4])

    colors = np.array([[1., 0., 0.], [0., 1., 0.]])
    colors_inner = np.array([[0., 0., 1.]])

    image = pad_image(image, verbose=1)
    image = tf.image.draw_bounding_boxes([image], boxes, colors)[0]
    image = tf.image.draw_bounding_boxes([image], boxes_inner, colors_inner)[0]

    return array_to_img(image)


def pad_and_crop_image(image: np.array):
    box_top_left_corners = get_bounding_boxes_start_point(image)
    padded_image = pad_image(image)

    cropped = []

    for x, y in zip(box_top_left_corners[0], box_top_left_corners[1]):
        cropped_image = tf.image.crop_to_bounding_box(padded_image, y, x, INPUT_SIZE, INPUT_SIZE)

        cropped.append(cropped_image)

    return cropped


def crop_data_samples(samples, masks):
    samples_cropped = []
    masks_cropped = []

    for s in samples:
        sample_cropped = pad_and_crop_image(s)
        samples_cropped.extend(sample_cropped)

    for m in masks:
        mask_cropped = pad_and_crop_image(m)
        masks_cropped.extend(mask_cropped)

    return samples_cropped, masks_cropped


def load_and_crop_training_data_samples():
    samples, masks = load_data_samples(TRAINING_DATA_DIRECTORY)

    return crop_data_samples(samples, masks)


def load_and_crop_validation_data_samples():
    samples, masks = load_data_samples(VALIDATION_DATA_DIRECTORY)

    return crop_data_samples(samples, masks)


# endregion


# region ROTATING INPUT IMAGES

def create_rotated_data_samples(samples, masks):
    samples_rotated = []
    masks_rotated = []

    for s in samples:
        samples_rotated.append(s)
        samples_rotated.append(tf.image.rot90(s))
        samples_rotated.append(tf.image.rot90(s, k=2))
        samples_rotated.append(tf.image.rot90(s, k=3))

    for m in masks:
        masks_rotated.append(m)
        masks_rotated.append(tf.image.rot90(m))
        masks_rotated.append(tf.image.rot90(m, k=2))
        masks_rotated.append(tf.image.rot90(m, k=3))

    return samples_rotated, masks_rotated


# endregion


# region MODEL

def double_conv_block(x, n_filters):
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)
    # Conv2D then ReLU activation
    x = layers.Conv2D(n_filters, 3, padding="same", activation="relu", kernel_initializer="he_normal")(x)

    return x


def downsample_block(x, n_filters):
    f = double_conv_block(x, n_filters)
    p = layers.MaxPool2D(2)(f)
    p = layers.Dropout(0.3)(p)

    return f, p


def upsample_block(x, conv_features, n_filters):
    # upsample
    x = layers.Conv2DTranspose(n_filters, 3, 2, padding="same")(x)
    # concatenate
    x = layers.concatenate([x, conv_features])
    # dropout
    x = layers.Dropout(0.3)(x)
    # Conv2D twice with ReLU activation
    x = double_conv_block(x, n_filters)

    return x


def prepare_unet_model(weight=1):
    # inputs
    inputs = layers.Input(shape=(INPUT_SIZE, INPUT_SIZE, 3))

    # encoder: contracting path - downsample
    # 1 - downsample
    f1, p1 = downsample_block(inputs, 64)
    # 2 - downsample
    f2, p2 = downsample_block(p1, 128)
    # 3 - downsample
    f3, p3 = downsample_block(p2, 256)
    # 4 - downsample
    f4, p4 = downsample_block(p3, 512)

    # 5 - bottleneck
    bottleneck = double_conv_block(p4, 1024)

    # decoder: expanding path - upsample
    # 6 - upsample
    u6 = upsample_block(bottleneck, f4, 512)
    # 7 - upsample
    u7 = upsample_block(u6, f3, 256)
    # 8 - upsample
    u8 = upsample_block(u7, f2, 128)
    # 9 - upsample
    u9 = upsample_block(u8, f1, 64)

    # outputs
    outputs = layers.Conv2D(2, 1, padding="same", activation="softmax")(u9)

    # unet model with Keras Functional API
    unet_model = Model(inputs, outputs, name="U-Net")

    unet_model.compile(optimizer=optimizers.Adam(),
                       loss="sparse_categorical_crossentropy",
                       loss_weights=(1, weight),
                       metrics=["accuracy", 
                                # metrics.MeanIoU(num_classes=2, sparse_y_true=False)
                               ])

    # unet_model.summary()

    return unet_model


# endregion


def train(unet_model: Model, samples, masks, validation_samples, validation_masks, epochs=20, batch_size=64):
    model_history = unet_model.fit(
        np.array([s.numpy() for s in samples]).reshape((len(samples), INPUT_SIZE, INPUT_SIZE, 3)),
        np.array([m.numpy() for m in masks]).reshape((len(masks), INPUT_SIZE, INPUT_SIZE, 1)),
        epochs=epochs,
        batch_size=batch_size,
        # validation_data=(
        #     np.array([s.numpy() for s in validation_samples]).reshape(
        #         (len(validation_samples), INPUT_SIZE, INPUT_SIZE, 3)),
        #     np.array([m.numpy() for m in validation_masks]).reshape(
        #         (len(validation_masks), INPUT_SIZE, INPUT_SIZE, 1))
        # ),
        # validation_batch_size=batch_size,
    )

    return model_history


def plot_training_progress(model_history):
    plt.plot(model_history.history['loss'], label="train")
    # plt.plot(model_history.history['val_loss'], label="validation")
    plt.title('Model Loss')
    plt.yscale('log')
    plt.ylabel('Cross Entropy')
    plt.xlabel('Iteration')
    plt.grid()
    plt.legend()
    plt.show()

    plt.plot(model_history.history['accuracy'], label="train")
    # plt.plot(model_history.history['val_accuracy'], label="validation")
    plt.title('Model Accuracy')
    plt.ylabel('Acc')
    plt.xlabel('Iteration')
    plt.grid()
    plt.legend()
    plt.show()


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
        plt.axis("off")
    plt.show()


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def show_prediction(unet_model: Model, sample, mask):
    pred_mask = unet_model.predict(sample.numpy().reshape((1, INPUT_SIZE, INPUT_SIZE, 3)))
    display([sample, mask, create_mask(pred_mask)])
    
    
def calculate_number_of_columns_and_rows(sample):
    box_top_left_corners = get_bounding_boxes_start_point(sample)
    columns_count, rows_count = len(np.unique(box_top_left_corners[0])), len(np.unique(box_top_left_corners[1]))
    
    return columns_count, rows_count
    

def show_full_image_prediction(unet_model: Model, sample, mask):    
    columns, rows = calculate_number_of_columns_and_rows(sample)
    
    samples_cropped, masks_cropped = pad_and_crop_image(sample), pad_and_crop_image(mask)
    
    masks_predicted = []
    for sample2 in samples_cropped:
        pred_mask = unet_model.predict(sample2.numpy().reshape((1, INPUT_SIZE, INPUT_SIZE, 3)), verbose=0)
        pred_mask = tf.image.crop_to_bounding_box(pred_mask, INPUT_MARGIN_SIZE, INPUT_MARGIN_SIZE, INPUT_CONTENT_SIZE, INPUT_CONTENT_SIZE)
        masks_predicted.append(pred_mask.numpy())
    
    c_list = []

    for i in range(rows):
        c = np.concatenate(masks_predicted[i * columns:(i + 1) * columns], axis=2)
        c_list.append(c)

    c = np.concatenate(c_list, axis=1)
    
    width, height = get_image_size(sample)
    width_pad_start, _ = calculate_padding_size(width)
    height_pad_start, _ = calculate_padding_size(height)
    
    c = tf.image.crop_to_bounding_box(c, height_pad_start - INPUT_MARGIN_SIZE, width_pad_start - INPUT_MARGIN_SIZE, height, width)
    
    print(c.shape, mask.shape)
    display([sample, mask, create_mask(c)])
    
def show_full_image_predictions(unet_model: Model, samples, masks):
    for sample, mask in zip(samples, masks):
        show_full_image_prediction(unet_model, sample, mask)


def show_predictions(unet_model: Model, samples, masks):
    for image, mask in zip(samples, masks):
        pred_mask = unet_model.predict(img_to_array(mask))
        display([image, mask, create_mask(pred_mask)])


def save_model(unet_model: Model):
    unet_model.save('model')


def load_model():
    return tf.keras.models.load_model('model')


def main():
    samples, masks = load_and_crop_training_data_samples()
    validation_samples, validation_masks = load_and_crop_validation_data_samples()
    samples, masks = create_rotated_data_samples(samples, masks)
    validation_samples, validation_masks = create_rotated_data_samples(validation_samples, validation_masks)

    unet_model = prepare_unet_model()

    model_history = train(unet_model, samples, masks, validation_samples, validation_masks, batch_size=8, epochs=20)

    save_model(unet_model)

    plot_training_progress(model_history)

    show_predictions(unet_model,
                     [samples[-40],
                      samples[-36],
                      samples[-68],
                      samples[-80]],
                     [masks[-40],
                      masks[-36],
                      masks[-68],
                      masks[-80]])

    return unet_model


if __name__ == "__main__":
    main()
