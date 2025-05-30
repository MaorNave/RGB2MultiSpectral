from back.DataGenerator_dev import DataGenerator


def create_data_for_annotations_from_input_sentilet_images():
    dg = DataGenerator()
    dg.create_data()

def create_masks_from_json_files():
    dg = DataGenerator()
    dg.create_masks_from_json_files()

def create_augmantations_for_masks():
    dg = DataGenerator()
    dg.create_augmantations_for_masks()

def move_candidates_randomly_to_ds():
    dg = DataGenerator()
    dg.move_candidates_to_dataset()

def check_for_k_cluster_number():
    dg = DataGenerator()
    dg.check_for_k_cluster_number()

def check_for_seg_classes():
    dg = DataGenerator()
    dg.check_for_seg_classes()





def main():


    """ - -----------------------------  DataGenerator images creation and augmantations - ----------------------------------------- """
    create_data_for_annotations_from_input_sentilet_images()

    # create_masks_from_json_files()

    # create_augmantations_for_masks()

    # move_candidates_randomly_to_ds()

    # check_for_k_cluster_number()

    # check_for_seg_classes()

if __name__ == "__main__":
    main()




