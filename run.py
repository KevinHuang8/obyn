import obyn.training.train_tree as t

if __name__ == '__main__':
    # Note: make sure to force reload whenever changing data size
    t.train(data_category='data_neon', force_reload=True, artificial_labels=False)
