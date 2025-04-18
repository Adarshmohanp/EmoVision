from tensorflow.keras.models import load_model, save_model

# Load your existing .keras model
model = load_model('newerfacemodel.keras')  # Change to your filename

# Save in .h5 format
save_model(model, 'model.h5')
print("✅ Model converted successfully to model.h5")