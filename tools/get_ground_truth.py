import scipy.io
import argparse

def load_mat_file(file_path):
    """
    Loads a .mat file and returns the data.

    :param file_path: Path to the .mat file.
    :return: The loaded data from the .mat file.
    """
    try:
        mat_data = scipy.io.loadmat(file_path)
        return mat_data
    except Exception as e:
        print(f"Error loading .mat file: {e}")
        return None


def extract_char_bound(mat_data, key_name):
    """
    Extracts the specified structure (trainCharBound or testCharBound) from the loaded .mat data.

    :param mat_data: Loaded .mat data.
    :param key_name: The key name to search for (trainCharBound or testCharBound).
    :return: The extracted structure or None if the key is not found.
    """
    try:
        char_bound_data = mat_data[key_name]
        return char_bound_data
    except KeyError:
        print(f"Error: '{key_name}' not found in the .mat file.")
        return None


def parse_char_bound(char_bound_data):
    """
    Extracts image names, characters, and bounding boxes from the charBound structure.

    :param char_bound_data: The charBound structure (either train or test).
    :return: Lists containing image names, characters, and bounding boxes.
    """
    image_names = []
    characters_list = []
    bounding_boxes = []

    for entry in char_bound_data[0]:
        image_name = entry['ImgName'][0]  # Image name
        characters = entry['chars'][0]  # Characters (string)
        char_bb = entry['charBB'][0]  # Bounding boxes for each character
        
        # Append extracted data to lists
        image_names.append(image_name)
        characters_list.append(characters)
        bounding_boxes.append(char_bb)

    return image_names, characters_list, bounding_boxes


def display_extracted_data(image_names, characters_list, bounding_boxes):
    """
    Displays the extracted data for each image in the structure.

    :param image_names: List of image names.
    :param characters_list: List of characters for each image.
    :param bounding_boxes: List of bounding boxes for each image.
    """
    for i in range(len(image_names)):
        print(f"Image Name: {image_names[i]}")
        print(f"Characters: {characters_list[i]}")
        print(f"Bounding Boxes: {bounding_boxes[i]}")
        print("-" * 30)  # Separator for readability


def main(file_path, key_name):
    """
    Main function to load the .mat file, extract data, and display it.

    :param file_path: Path to the .mat file.
    :param key_name: The key name to extract ('trainCharBound' or 'testCharBound').
    """
    # Load the .mat file
    mat_data = load_mat_file(file_path)
    if mat_data is None:
        return

    # Extract the specified charBound structure (train or test)
    char_bound_data = extract_char_bound(mat_data, key_name)
    if char_bound_data is None:
        return

    # Extract image names, characters, and bounding boxes
    image_names, characters_list, bounding_boxes = parse_char_bound(char_bound_data)

    # Display extracted data
    display_extracted_data(image_names, characters_list, bounding_boxes)


if __name__ == "__main__":
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Parse and extract data from a .mat file containing charBound.")
    parser.add_argument("--file_path", help="Path to the .mat file containing the charBound data.")
    parser.add_argument("--key_name", choices=['trainCharBound', 'testCharBound'], help="Specify 'trainCharBound' or 'testCharBound' to extract the respective structure.")
    
    # Parse arguments from the terminal
    args = parser.parse_args()

    # Call main function with the provided file path and key name
    main(args.file_path, args.key_name)
