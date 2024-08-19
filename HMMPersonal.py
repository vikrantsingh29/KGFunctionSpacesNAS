import os

# Set the directory path
dir_path = r'C:\Users\vikrant.singh\Desktop\HMMData\data_vikrant'

# Initialize an empty string to hold the concatenated text
concatenated_text = ''

# Loop through all files in the directory
for file_name in os.listdir(dir_path):

    # Check if the file is a text file
    if file_name.endswith('.txt'):
        # Open the file and read the text
        with open(os.path.join(dir_path, file_name), 'r') as f:
            file_text = f.read()

        # Concatenate the text to the existing string
        concatenated_text += file_text

# Write the concatenated text to a new file
with open(os.path.join(dir_path, 'concatenated_text.txt'), 'w') as f:
    f.write(concatenated_text)
