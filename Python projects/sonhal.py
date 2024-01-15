import pandas as pd
import cv2
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

# Step 1: Data Collection
# Assuming you have a DataFrame 'data' with columns: 'image_path', 'label', 'distance'
# Manually create or load your dataset
data = pd.DataFrame({
    'image_path': ['path/to/image1.jpg', 'path/to/image2.jpg', ...],
    'label': [0, 1, ...],  # 0 for heads, 1 for tails
    'distance': [1.5, 2.0, ...],  # distances from the origin
})

# Step 2: Data Labeling (Already labeled in the dataset)

# Step 3: Preprocessing
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0  # Normalize pixel values
    return img

# Extract features and labels from the dataset
def extract_features_labels(data):
    images = []
    labels = []

    for index, row in data.iterrows():
        img_path = row['image_path']
        label = row['label']

        # Preprocess the image
        img = preprocess_image(img_path)

        # Append the preprocessed image and label to the lists
        images.append(img)
        labels.append(label)

    # Convert lists to numpy arrays
    images = np.array(images)
    labels = np.array(labels)

    return images, labels

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Extract features and labels
X, y = extract_features_labels(data)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Model Selection
input_shape = (224, 224, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Conv2D(128, (3, 3), activation='relu'))
model.add(MaxPooling2D((2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Step 5: Model Training
history = model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

# Step 6: Model Evaluation
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}, Test Accuracy: {test_accuracy}')

# Step 7: Video Validation
cap = cv2.VideoCapture('path/to/your/video.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_image(frame)

    # Reshape the frame to match the input shape expected by the model
    processed_frame = np.reshape(processed_frame, (1,) + processed_frame.shape)

    # Make predictions
    prediction = model.predict(processed_frame)

    # Convert probability to binary prediction
    prediction_binary = (prediction > 0.5).astype(int)

    # Display the frame with prediction
    if prediction_binary == 0:
        cv2.putText(frame, 'Heads', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        cv2.putText(frame, 'Tails', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)

    cv2.imshow('Coin Drop Prediction', frame)

    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

