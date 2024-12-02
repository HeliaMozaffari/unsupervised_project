from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
import math
import tensorflow as tf
import numpy as np
import random
from collections import Counter
from collections import defaultdict

file_path = 'umist_cropped.mat'
data = loadmat(file_path)

images = data['facedat'][0]
labels = data['dirnames'][0]

all_images = []
all_labels = []

for i, person_images in enumerate(images):
    label = labels[i][0]
    for img_idx in range(person_images.shape[2]):
        img = person_images[:, :, img_idx]
        all_images.append(img.flatten())
        all_labels.append(f"{label}_{img_idx}")
        
df = pd.DataFrame({'label': all_labels, 'image': all_images})
df['image'] = df['image'].apply(lambda x: tuple(x))

duplicates = df[df.duplicated(subset='image', keep=False)]
unique_df = df.drop_duplicates(subset='image').reset_index(drop=True)

print(f"Original number of images: {len(df)}")
print(f"Number of duplicates rows: {len(duplicates)}")
print(f"Final number of unique images: {len(unique_df)}")

unique_df['label'] = unique_df['label'].str.split('_').str[0]
print(unique_df.head())

'''
for label in unique_df['label'].unique():
    label_group = unique_df[unique_df['label'] == label]
    num_images = len(label_group)

    cols = 5
    rows = math.ceil(num_images / cols)

    fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 3))
    fig.suptitle(f'Label: {label}', fontsize=16)

    for i, (_, row) in enumerate(label_group.iterrows()):
        ax = axs[i // cols, i % cols]
        img = np.array(row['image']).reshape(112, 92)
        ax.imshow(img, cmap='gray')
        ax.set_title(f'Image {i+1}')
        ax.axis('off')

    for j in range(i + 1, rows * cols):
        axs[j // cols, j % cols].axis('off')

    plt.show()
    
'''
    
'''Augment a grayscale image using random shift and zoom with fixed seed.'''
def augment_image(image):
    if not isinstance(image, tf.Tensor):
        image = tf.convert_to_tensor(image, dtype=tf.float32)

    tf.random.set_seed(None)
    shift = tf.random.uniform([], -0.05, 0.05)
    image = tf.keras.preprocessing.image.apply_affine_transform(
        image.numpy(),
        tx=shift * image.shape[0],
        ty=shift * image.shape[1],
        fill_mode='nearest'
    )

    zoom = tf.random.uniform([], 0.98, 1.02)
    image = tf.keras.preprocessing.image.apply_affine_transform(
        image,
        zx=zoom,
        zy=zoom,
        fill_mode='nearest'
    )

    return tf.convert_to_tensor(image, dtype=tf.float32)

augmented_images = []
augmented_labels = []

target_count = unique_df['label'].value_counts().max()
images_by_label = unique_df.groupby('label')['image'].apply(list).to_dict()

for label, images in images_by_label.items():
    current_count = len(images)

    if current_count < target_count:
        needed = target_count - current_count

        for _ in range(needed):
            image_index = random.randint(0, current_count - 1)
            image_to_augment = np.array(images[image_index]).reshape(112, 92, 1)

            image_tensor = tf.convert_to_tensor(image_to_augment, dtype=tf.float32)
            augmented_image = augment_image(image_tensor)

            augmented_images.append(augmented_image.numpy().squeeze())
            augmented_labels.append(label)
            
combined_images = []
combined_labels = []

for _, row in unique_df.iterrows():
    combined_images.append(np.array(row['image']).reshape(112, 92))
    combined_labels.append(row['label'])


combined_images.extend(augmented_images)
combined_labels.extend(augmented_labels)

combined_images_flat = [tuple(image.flatten()) for image in combined_images]

df_after_aug = pd.DataFrame({
    'label': combined_labels,
    'image': combined_images_flat
})

duplicates = df_after_aug[df_after_aug.duplicated(subset='image', keep=False)]

print(f"Total images: {len(df_after_aug)}")
print(f"Number of duplicates: {len(duplicates)}")
print(f"Number of unique images: {len(df_after_aug.drop_duplicates(subset='image'))}")

Counter(combined_labels)

images_by_label = defaultdict(list)

for img, label in zip(combined_images, combined_labels):
    images_by_label[label].append(img)


# for label, images in images_by_label.items():
#     num_images = len(images)
#     cols = 5
#     rows = math.ceil(num_images / cols)

#     fig, axs = plt.subplots(rows, cols, figsize=(15, rows * 3))
#     fig.suptitle(f"Label: {label}", fontsize=16)

#     for i in range(rows * cols):
#         ax = axs[i // cols, i % cols]
#         if i < num_images:
#             ax.imshow(images[i], cmap='gray')
#             ax.set_title(f"{label} - Image {i+1}")
#             ax.axis('off')
#         else:
#             ax.axis('off')

#     plt.tight_layout()
#     plt.subplots_adjust(top=0.9)
#     plt.show()
#     plt.close(fig)

'''
Noramlize
'''
for label, images in images_by_label.items():
    images_by_label[label] = [img / 255.0 for img in images]
    
'''
Train, Validation, Test Split
'''
train_count = 40
validation_count = 0
test_count = 8

train_images, train_labels = [], []
validation_images, validation_labels = [], []
test_images, test_labels = [], []

for label, images in images_by_label.items():
    if len(images) < train_count + validation_count + test_count:
        print(f"Skipping label {label}: insufficient images ({len(images)})")
        continue

    np.random.shuffle(images)

    train_images.extend(images[:train_count])
    train_labels.extend([label] * train_count)

    validation_images.extend(images[train_count:train_count + validation_count])
    validation_labels.extend([label] * validation_count)

    test_images.extend(images[train_count + validation_count:train_count + validation_count + test_count])
    test_labels.extend([label] * test_count)
    
print(len(train_images), len(validation_images), len(test_images))
