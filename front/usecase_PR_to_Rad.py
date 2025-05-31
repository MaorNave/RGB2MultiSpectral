from front.Materials import Materials
from front.Materials_prossesor import Materials_prossesor
from front.Materials_tests import Materials_test


def cubes_generator():
    materials = Materials()
    materials.cubes_generator()

def cubes_generator_raffle():
    materials = Materials()
    materials.cubes_generator_raffle()


def heuristic_rgb_material_vector_v1():
    materials = Materials_prossesor()
    materials.heuristic_rgb_material_vector_v1()

def heuristic_object_material_vector():
    materials = Materials_prossesor()
    materials.heuristic_object_material_vector()

def vgg19_clasiffier():
    materials = Materials_prossesor()
    materials.vgg19_classifier()


def plot_multispectral_rgb_and_texture():
    materials = Materials_test()
    materials.plot_multispectral_rgb_and_texture()

def compare_metrics():
    materials = Materials_test()
    materials.compare_metrics()


def compare_materials_signatures():
    materials = Materials_test()
    materials.compare_materials_signatures()

def compare_metrics_rgb():
    materials = Materials_test()
    materials.compare_metrics_rgb()


def main():
    """ ------------------------------  PR_TO_RAD PREPROSSES PIPELINE ------------------------------------------ """

    # heuristic_object_material_vector()
    # vgg19_classifier()
    # heuristic_rgb_material_vector_v1()

    """ ------------------------------  PR_TO_RAD CUBES GENERATOR PIPELINE ------------------------------------------ """
    # config = Materials.yaml_loader("config_PRTORAD.yaml")
    # implementation_method = config['cubes_generator']['transplant_method']
    # if implementation_method == 'max_score':
    #     cubes_generator()
    # elif implementation_method == 'raffle':
    #     cubes_generator_raffle()

    """ ------------------------------  PR_TO_RAD CUBES TEST DATA ------------------------------------------ """
    # plot_multispectral_rgb_and_texture()
    # compare_materials_signatures()
    # compare_metrics()
    # compare_metrics_rgb()



if __name__ == "__main__":
    main()