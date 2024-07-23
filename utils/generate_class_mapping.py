import os

def generate_class_mapping(dataset_path):
    class_mapping = {}
    class_id = 0

    # Read directory names
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    print(f"Class directories found: {class_names}")
    
    for class_name in class_names:
        class_dir = os.path.join(dataset_path, class_name)
        if os.path.isdir(class_dir):
            print(f"Mapping class ID {class_id} to class name '{class_name}'")
            class_mapping[class_id] = class_name
            class_id += 1

    return class_mapping

def save_class_mapping(class_mapping, output_path):
    with open(output_path, 'w') as f:
        f.write("class_mapping = {\n")
        for class_id, class_name in class_mapping.items():
            f.write(f"    {class_id}: '{class_name}',\n")
        f.write("}\n")

def validate_class_mapping(class_mapping, dataset_path):
    class_names = sorted([d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))])
    mapping_class_names = [class_mapping[i] for i in range(len(class_mapping))]

    print("Classes in dataset folders:")
    print(class_names)
    print("\nClasses in generated mapping:")
    print(mapping_class_names)

    if class_names == mapping_class_names:
        print("\nValidation successful: The class mapping matches the dataset structure.")
    else:
        print("\nValidation failed: The class mapping does not match the dataset structure.")
        print("Differences:")
        for i, (expected, actual) in enumerate(zip(class_names, mapping_class_names)):
            if expected != actual:
                print(f"  - At index {i}: Expected '{expected}', but got '{actual}'")

if __name__ == "__main__":
    dataset_path = '/data/train/'  # Update this path to your dataset train directory
    output_path = 'class_mapping.py'
    
    class_mapping = generate_class_mapping(dataset_path)
    save_class_mapping(class_mapping, output_path)
    print(f"Class mapping saved to {output_path}")

    # Validate the generated class mapping
    validate_class_mapping(class_mapping, dataset_path)
