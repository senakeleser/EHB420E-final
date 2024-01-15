# X_new contains the preprocessed images of new coin drops

# Predict probabilities
predictions_probabilities = model.predict(X_new)

# Convert probabilities to binary predictions (0 for heads, 1 for tails)
predictions_binary = (predictions_probabilities > 0.5).astype(int)

# Display the predictions
for i, pred in enumerate(predictions_binary):
    if pred == 0:
        print(f'Prediction {i + 1}: Heads')
    else:
        print(f'Prediction {i + 1}: Tails')
