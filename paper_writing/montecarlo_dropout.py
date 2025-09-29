# 3 Uncertainty Quantification via Monte Carlo Dropout - Python Code (TensorFlow/Keras)
#  This requires that your model was trained with Dropout layers and that they are still active during inference.
# Step 1: Modify Your Model for MC Dropout
#   Ensure your model has dropout layers. The one you described (dropout rate of 0.2 before the final layer) is perfect.
# Step 2: Code for MC Dropout Sampling

import numpy as np
import tensorflow as tf

def mc_dropout_predict(model, input_data, n_samples=50):
    """
    Performs Monte Carlo Dropout sampling to get multiple predictions.
    
    Args:
        model (tf.keras.Model): The trained model with Dropout layers.
        input_data (numpy.ndarray): Input image batch (shape: [1, D, H, W, C]).
        n_samples (int): Number of stochastic forward passes.
    
    Returns:
        numpy.ndarray: Array of probabilistic predictions [n_samples, D, H, W, num_classes]
        numpy.ndarray: Mean probability across samples [D, H, W, num_classes]
        numpy.ndarray: Standard deviation (uncertainty) across samples [D, H, W, num_classes]
    """
    # This function must be called with training=True to enable dropout!
    # Keras assumes training=False during standard model.predict(), which disables dropout.
    predict_stochastic = tf.function(lambda x: model(x, training=True))
    
    # Run predictions multiple times
    predictions = []
    for _ in range(n_samples):
        prob_pred = predict_stochastic(input_data).numpy()
        predictions.append(prob_pred)
    
    # Stack the predictions
    stacked_preds = np.stack(predictions, axis=0) # Shape: [n_samples, 1, D, H, W, C]
    stacked_preds = np.squeeze(stacked_preds, axis=1) # Remove batch dim -> [n_samples, D, H, W, C]
    
    # Calculate mean and standard deviation (uncertainty) across the samples
    mean_prob = np.mean(stacked_preds, axis=0)
    std_prob = np.std(stacked_preds, axis=0) # This is the voxel-wise uncertainty map
    
    return stacked_preds, mean_prob, std_prob

# Example usage:
# input_image = test_image[np.newaxis, ...] # Add batch dimension: [1, 80, 80, 96, 1]
# n_samples = 25
# all_samples, mean_pred, uncertainty_map = mc_dropout_predict(your_trained_model, input_image, n_samples)

# Get the final, robust segmentation from the mean probabilities
# final_segmentation = np.argmax(mean_pred, axis=-1) # Shape: [D, H, W]

# The 'uncertainty_map' has shape [D, H, W, num_classes]. You often want the uncertainty
# for the predicted class. Here's how to extract it:
# predicted_class = final_segmentation
# uncertainty_for_prediction = np.zeros_like(predicted_class, dtype=np.float32)
# for i in range(num_classes):
#     mask = (predicted_class == i)
#     uncertainty_for_prediction[mask] = uncertainty_map[..., i][mask]

# Step 3: Visualize the Uncertainty
# Plot the uncertainty for a specific slice
slice_to_show = 40
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.imshow(test_image[slice_to_show, :, :, 0], cmap='gray')
plt.contour(final_segmentation[slice_to_show], colors='r')
plt.title('Final Segmentation')
plt.axis('off')

plt.subplot(1, 2, 2)
im = plt.imshow(uncertainty_for_prediction[slice_to_show], cmap='hot')
plt.colorbar(im, label='Uncertainty (Std Dev)')
plt.title('Voxel-wise Predictive Uncertainty')
plt.axis('off')

plt.tight_layout()
plt.show()

#What to Report:
#  Qualitatively: Show an image of the uncertainty map overlayed on the anatomy. You will almost certainly see that the boundaries between brainstem and CSF, and between the sub-regions, have the highest uncertainty. This makes intuitive sense and validates the method.
#  Quantitatively: You can calculate the average uncertainty within the segmented region. You could also correlate Dice score with average uncertainty across your test set: poor-performing cases (low Dice) should have higher average uncertainty. Finding this correlation would be a very strong result.