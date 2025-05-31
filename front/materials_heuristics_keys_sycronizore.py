

def keys_sycronizore(data):
    new_keys_list = []
    for key in data.keys():
        if 'water' in key.lower():
            key += ';waters_and_snow'
            new_keys_list.append(key)
        elif 'lands_compounds' in key.lower():
            key += ';lands_compounds_soils_and_sands'
            new_keys_list.append(key)
        elif 'construction_materials' in key.lower():
            key += ';lands_compounds_construction_materials'
            new_keys_list.append(key)
        elif 'vegetation' in key.lower():
            key += ';vegetation'
            new_keys_list.append(key)
        elif 'road' in key.lower():
            key += ';roads_materials'
            new_keys_list.append(key)
        elif 'soil' in key.lower():
            key += ';soils_and_sands'
            new_keys_list.append(key)
    return new_keys_list
