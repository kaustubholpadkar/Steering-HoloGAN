from auregressor import ActionUnitRegressor

aur = ActionUnitRegressor(csv_dir="EmotionNet/imgs_GANimationCrop_aus", img_dir="EmotionNet/imgs_GANimationCrop")

aur.train_model()
