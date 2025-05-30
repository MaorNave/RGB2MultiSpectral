from back.NN_dev import vgg_train_val_session, vgg_test_session


def train_val_session(kwargs):
    vgg_train_val_session(**kwargs)

def test_session(kwargs):
    vgg_test_session(**kwargs)








def main():


    """ - -----------------------------  NN_dev train_val_session - ----------------------------------------- """
    # kwargs = {'main_path': "C:\\Maor Nanikashvili\\thesis_pipeline\Pipeline\\results\\sentilet_data\\full_dataset",
    # 'train_batch_size': 32,
    # 'val_batch_size': 16,
    # 'num_classes': 6,
    # 'lr': 0.0001,
    # 'num_epochs': 1000
    # }
    #
    # train_val_session(kwargs)

    """ - -----------------------------  NN_dev infer on test_data - ----------------------------------------- """

    # kwargs = {'num_classes': 6,
    #          'path_to_trained_weights': "C:\Maor Nanikashvili\\thesis_pipeline\Pipeline\input\\net_weights\\fab_25_session_with_self_annotations\\new_model_weights_lr_0.0001_775.pth",
    #          'input_image_path': "C:\Maor Nanikashvili\\thesis_pipeline\Pipeline\\results\sentilet_data\\full_dataset\\test\images\T36RXV_20240613T081611_20m_patch_152_15.jpg",
    #          'gt_mask_path': "C:\Maor Nanikashvili\\thesis_pipeline\Pipeline\\results\sentilet_data\\full_dataset\\test\masks\T36RXV_20240613T081611_20m_patch_152_15.png"}
    #
    # test_session(kwargs)

if __name__ == "__main__":
    main()




