class Config:
    # для воспроизводимости
    SEED = 42

    # Модели
    TEXT_MODEL_NAME = 'distilbert-base-uncased'
    IMAGE_MODEL_NAME =  'tf_efficientnetv2_s'

    # Какие слои размораживаем - совпадают с нэймингом в моделях
    TEXT_MODEL_UNFREEZE = "transformer.layer.4|transformer.layer.5"
    IMAGE_MODEL_UNFREEZE = "blocks.4|blocks.5|conv_head|bn2"

    # Гиперпараметры
    BATCH_SIZE = 32
    FINE_TUNE_BATCH_SIZE = 16
    EPOCHS = 100
    
    TEXT_LR = 2e-5
    IMAGE_LR = 1e-5
    MASS_PROJECTOR_LR = 1e-3
    REGRESSOR_LR = 1e-3
    
    DROPOUT = 0.3
    
    MASS_PROJECTION_DIM = 32

    # Пути
    DISHES_DF_PATH = "data/dish.csv"
    INGRIDIENTS_DF_PATH = "data/ingredients.csv"
    SAVE_PATH = "models/best_model.pth"