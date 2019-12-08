from auregressor import ActionUnitRegressor


csv_dir = "./EmotionNet/imgs_GANimationCrop_aus"
img_dir = "./EmotionNet/imgs_GANimationCrop"

lr = 0.001
momentum = 0.9
batch_size = 128

aur = ActionUnitRegressor(
    csv_dir, img_dir, lr, momentum, batch_size
)

epochs = 10

aur.train_model(num_epochs=epochs)

save_file = "./trained_aur.pkl"

aur.save(save_file, epochs)
