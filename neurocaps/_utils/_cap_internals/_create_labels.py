"""Internal function to create labels at the node level for caps2plot"""
import collections, re

def _create_node_labels(parcellation_name, parcel_approach, columns):
    # Get frequency of each major hemisphere and region in Schaefer, AAL, or Custom atlas
    if parcellation_name == "Schaefer":
        nodes = parcel_approach[parcellation_name]["nodes"]
        # Retain only the hemisphere and primary Schaefer network
        nodes = [node.split("_")[:2] for node in nodes]
        frequency_dict = collections.Counter([" ".join(node) for node in nodes])
    elif parcellation_name == "AAL":
        nodes = parcel_approach[parcellation_name]["nodes"]
        frequency_dict = collections.Counter([node.split("_")[0] for node in nodes])
    else:
        frequency_dict = {}
        for names_id in columns:
            # For custom, columns comes in the form of "Hemisphere Region"
            hemisphere_id = "LH" if names_id.startswith("LH ") else "RH"
            region_id = re.split("LH |RH ", names_id)[-1]
            node_indices = parcel_approach["Custom"]["regions"][region_id][hemisphere_id.lower()]
            frequency_dict.update({names_id: len(node_indices)})

    # Get the names, which indicate the hemisphere and region
    # Reverting Counter objects to list retains original ordering of nodes in list as of Python 3.7
    names_list = list(frequency_dict)
    labels = ["" for _ in range(0,len(parcel_approach[parcellation_name]["nodes"]))]

    starting_value = 0

    # Iterate through names_list and assign the starting indices corresponding to unique region and hemisphere key
    for num, name in enumerate(names_list):
        if num == 0:
            labels[0] = name
        else:
            # Shifting to previous frequency of the preceding network to obtain the new starting value of
            # the subsequent region and hemisphere pair
            starting_value += frequency_dict[names_list[num-1]]
            labels[starting_value] = name

    return labels, names_list
