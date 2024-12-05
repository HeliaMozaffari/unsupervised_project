"""
Group 5
"""

from scipy.io import loadmat
import pandas as pd
import matplotlib.pyplot as plt
# import math
import tensorflow as tf
import numpy as np
import random
from collections import Counter
from collections import defaultdict

file_path = r"project\umist_cropped.mat"
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
        
sample_image = images[0][:, :, 0]
height, width = sample_image.shape

df = pd.DataFrame({'label': all_labels, 'image': all_images})
df['image'] = df['image'].apply(lambda x: tuple(x))

duplicates = df[df.duplicated(subset='image', keep=False)]
unique_df = df.drop_duplicates(subset='image').reset_index(drop=True)

print(f"Original number of images: {len(df)}")
print(f"Number of duplicates rows: {len(duplicates)}")
print(f"Final number of unique images: {len(unique_df)}")

unique_df['label'] = unique_df['label'].str.split('_').str[0]
print(unique_df.head())

unique_df.shape
    
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
print(df_after_aug.info())

images_by_label = defaultdict(list)

'''
Normalize
'''
df_after_aug['image'] = df_after_aug['image'].apply(lambda img: [pixel / 255.0 for pixel in img])


print(df_after_aug.head())
print(df_after_aug.shape)

#=================================
# KMeans Clustering
#=================================

'''
Train, Validation, Test Split with stratified
'''

from sklearn.model_selection import StratifiedShuffleSplit
X = np.array(df_after_aug['image'].tolist())
y = np.array(df_after_aug['label'].tolist())

split1 = StratifiedShuffleSplit(test_size=0.3, random_state=35)
for train_index, test_index in split1.split(X, y):
    X_train, X_temp = X[train_index], X[test_index]
    y_train, y_temp = y[train_index], y[test_index]
    
split2 = StratifiedShuffleSplit(test_size=0.5, random_state=35)
for test_data, val_data in split2.split(X_temp, y_temp):
    X_val, X_test = X[test_data], X[val_data]
    y_val, y_test = y[test_data], y[val_data]

print(f"Train X Data Shape: {X_train.shape}")
print(f"Train y Data Shape: {y_train.shape}")
print(f"Val X Data Shape: {X_val.shape}")
print(f"Val y Data Shape: {y_val.shape}")
print(f"Test X Data Shape: {X_test.shape}")
print(f"Test y Data Shape: {y_test.shape}")

"""
Extra Pre Processing
"""
#Standardize the Data: As KMeans uses Euclidean distance, crucial to standardize the data to ensure all features contribute equally.
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler() 
train_images_scaled = scaler.fit_transform(X_train) 
val_images_scaled = scaler.transform(X_val)
test_images_scaled = scaler.transform(X_test)

#Reduce dimensions using PCA
from sklearn.decomposition import PCA

pca = PCA(n_components=0.99, random_state=32)
X_train_pca = pca.fit_transform(train_images_scaled)
X_val_pca = pca.transform(val_images_scaled)
X_test_pca = pca.transform(test_images_scaled)

print("X Train Shape after pca:", X_train_pca.shape)
print("X Val Shape after pca:", X_test_pca.shape)
print("X Test Shape after pca:", X_test_pca.shape)


"""
Identify best K
"""
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
silhouette_scores = []
k_values = range(3, 29, 3)
for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=32)
    kmeans.fit(X_train_pca)
    silhouette_scores.append(silhouette_score(X_train_pca, kmeans.labels_))

# Find maximum silhouette score
best_index = np.argmax(silhouette_scores)
best_k = k_values[best_index] 
print(f"{'='*20}\nOptimal K: {best_k}")

# Visualize Silhouette Scores
plt.figure(figsize=(10, 6))
plt.plot(k_values, silhouette_scores, marker='X')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('Silhouette Score')
plt.show()

"""
Kmeans algorithm
"""

kmeans = KMeans(n_clusters=best_k, random_state=32)

X_train_reduced = kmeans.fit_transform(X_train_pca)
X_val_reduced = kmeans.transform(X_val_pca)
X_test_reduced = kmeans.transform(X_test_pca)

X_train_kmeans_labels = kmeans.fit_predict(X_train_pca)
X_val_kmeans_labels = kmeans.predict(X_val_pca)
X_test_kmeans_labels = kmeans.predict(X_test_pca)

cluster_labels = kmeans.labels_
cluster_centers = kmeans.cluster_centers_

# Plot the Clusters with their centers
plt.scatter(X_train_pca[:, 0], X_train_pca[:, 1], c=cluster_labels, cmap='winter', alpha=0.5)
plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], c='red', marker='*', s=20, label='Cluster Centers')
plt.title('KMeans Clustering')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()

print("="*30)
print("Shape of X_train_reduced:", X_train_reduced.shape)
print("Shape of X_val_reduced:", X_val_reduced.shape)
print("Shape of X_test_reduced:", X_test_reduced.shape)


# Plot the images that are present in each cluster
def plot_images_in_clusters(images, labels, cluster_labels, num_clusters, height, width):
    fig, axes = plt.subplots(num_clusters, 10, figsize=(20, 2 * num_clusters))
    for cluster in range(num_clusters):
        cluster_indices = np.where(cluster_labels == cluster)[0]
        selected_indices = np.random.choice(cluster_indices, 10, replace=False)
        for i, idx in enumerate(selected_indices):
            ax = axes[cluster, i]
            ax.imshow(images[idx].reshape(height, width), cmap='gray')
            ax.axis('off')
            ax.set_title(f"Cluster {cluster}")
    plt.tight_layout()
    plt.show()

plot_images_in_clusters(combined_images, combined_labels, X_train_kmeans_labels, best_k, height, width)

'''
4. Discuss the architecture your team has selected for training and predicting the test instances. 
'''
from tensorflow import keras
from tensorflow.keras import layers

# Concatenate cluster labels with flattened image data
X_train_combined = np.concatenate((X_train_pca, X_train_reduced), axis=1)
X_val_combined = np.concatenate((X_val_pca, X_val_reduced), axis=1)
X_test_combined = np.concatenate((X_test_pca, X_test_reduced), axis=1)

model = keras.Sequential([
    layers.Input(shape=(X_train_combined.shape[1],)),
    layers.Dense(128, activation='relu'),
    layers.Dense(64, activation='relu'),
    layers.Dense(best_k, activation='softmax')
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.summary()

"""
Results
"""
model.fit(X_train_combined, X_train_kmeans_labels, epochs=100, validation_data=(X_val_combined, X_val_kmeans_labels))

validation_accuracy = model.evaluate(X_val_combined, X_val_kmeans_labels)[1]
test_accuracy = model.evaluate(X_test_combined, X_test_kmeans_labels)[1]

print("Validation Accuracy:", validation_accuracy)
print("Test Accuracy:", test_accuracy)

predictions = model.predict(X_val_combined)
predicted_clusters_valid = np.argmax(predictions, axis=1)

def plot_original_and_predicted_images(original_images, predicted_images, original_labels, predicted_labels, indices, height, width):
    fig, axes = plt.subplots(5, 2, figsize=(10, 25))
    
    for i, index in enumerate(indices):
        # Original image
        axes[i, 0].imshow(original_images[index].reshape(height, width), cmap='gray')
        axes[i, 0].set_title(f"Original: {original_labels[index]}")
        axes[i, 0].axis('off')
        
        # Predicted image
        axes[i, 1].imshow(predicted_images[index].reshape(height, width), cmap='gray')
        axes[i, 1].set_title(f"Predicted: {predicted_labels[index]}")
        axes[i, 1].axis('off')
    
    plt.tight_layout()
    plt.show()

indices = random.sample(range(len(X_val_combined)), 5)
plot_original_and_predicted_images(X_val, X_val, y_val, predicted_clusters_valid, indices, height, width)