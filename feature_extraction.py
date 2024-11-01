import pefile
import math

def calculate_entropy(data):
    if not data:
        return 0
    entropy = 0
    for x in range(256):
        p_x = float(data.count(x)) / len(data)
        if p_x > 0:
            entropy += - p_x * math.log(p_x, 2)
    return entropy

def extract_info(file_path):
    pe = pefile.PE(file_path)


    image_base = pe.OPTIONAL_HEADER.ImageBase
    version_info_size = pe.OPTIONAL_HEADER.DATA_DIRECTORY[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_RESOURCE']].Size
    sections_max_entropy = max(section.get_entropy() for section in pe.sections)
    major_os_version = pe.OPTIONAL_HEADER.MajorOperatingSystemVersion
    size_of_stack_reserve = pe.OPTIONAL_HEADER.SizeOfStackReserve
    characteristics = pe.FILE_HEADER.Characteristics
    size_of_initialized_data = pe.OPTIONAL_HEADER.SizeOfInitializedData
    major_subsystem_version = pe.OPTIONAL_HEADER.MajorSubsystemVersion
    subsystem = pe.OPTIONAL_HEADER.Subsystem
    base_of_data = pe.OPTIONAL_HEADER.BaseOfData
    size_of_image = pe.OPTIONAL_HEADER.SizeOfImage
    major_linker_version = pe.OPTIONAL_HEADER.MajorLinkerVersion


    resources_min_entropy = float('inf') 
    resources_min_size = None  
    resources_nb = 0  
    for resource_type in pe.DIRECTORY_ENTRY_RESOURCE.entries:
        for resource_id in resource_type.directory.entries:
            if hasattr(resource_id, 'directory'):
                for resource_lang in resource_id.directory.entries:
                    lang_data = pe.get_data(resource_lang.data.struct.OffsetToData, resource_lang.data.struct.Size)
                    entropy = calculate_entropy(lang_data)
                    resources_min_entropy = min(resources_min_entropy, entropy)
            else:
                lang_data = pe.get_data(resource_id.data.struct.OffsetToData, resource_id.data.struct.Size)
                size = resource_id.data.struct.Size
                resources_nb += 1
                if resources_min_size is None:
                    resources_min_size = size
                else:
                    resources_min_size = min(resources_min_size, size)

   
    pe.close()

    return {
        "Image Base": image_base,
        "Version Information Size": version_info_size,
        "Sections Max Entropy": sections_max_entropy,
        "Major Operating System Version": major_os_version,
        "Size of Stack Reserve": size_of_stack_reserve,
        "Characteristics": characteristics,
        "Size of Initialized Data": size_of_initialized_data,
        "Major Subsystem Version": major_subsystem_version,
        "Subsystem": subsystem,
        "Base of Data": base_of_data,
        "Size of Image": size_of_image,
        "Major Linker Version": major_linker_version,
        "Resources Min Entropy": resources_min_entropy,
        "Resource Min Size": resources_min_size if resources_min_size is not None else 0,
        "Resource NB": resources_nb
    }


file_path =r"C:\Users\Python_38\Downloads\iconsext\iconsext.exe"
info = extract_info(file_path)
for key, value in info.items():
    print(f"{key}: {value}")
