import cv2
import numpy as np

# Open the video capture
cap = cv2.VideoCapture('drop0001.mp4')

while cap.isOpened():
    ret, frame = cap.read()

    if not ret:
        break

    # Preprocess the frame
    processed_frame = preprocess_image(frame)  # Use the same preprocessing function as during training

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
