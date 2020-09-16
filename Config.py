Config = {
    'train_size': 227,     # 图像尺寸均值
    'train_batch_size': 128,
    'test_batch_size': 128,
    'epochs': 100,
    'lr': 0.001,
    'momentum': 0.9,
    'phase': 'train',
    'model_dir': 'trained_model',
    'positive_weight': 1,
    'negative_weight': 3,
    'best_wts_name': 'best_wts.pt',
}