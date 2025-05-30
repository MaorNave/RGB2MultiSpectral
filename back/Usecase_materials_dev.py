from back.Materials_dev import Materials_dev


def raw_asd_to_yaml():
    materials = Materials_dev()
    materials.raw_asd_to_yaml()


def raw_spectral_lib_to_yaml():
    materials = Materials_dev()
    materials.raw_spectral_lib_to_yaml()

def classes_adder():
    materials = Materials_dev()
    materials.classes_adder()

def generate_materials_df():
    materials = Materials_dev()
    materials.generate_materials_df()



def main():


    """ - -----------------------------  Materials_dev data manipulations - ----------------------------------------- """
    # raw_asd_to_yaml()
    # raw_spectral_lib_to_yaml()
    # classes_adder()
    # generate_materials_df()

if __name__ == "__main__":
    main()




