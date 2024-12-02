import os

def create_project_structure(base_path = 'Classification project'):
          #Define the structure
          structure = [
                  'data',
                  'src',
                  'reports',
                  'requirements.txt',
                  'README.md']
          
          if not os.path.exists(base_path):
                  os.makedirs(base_path)
                  print(f'Created base directory:{base_path}')
          else:
                  print(f'Base directory already exists:{base_path}')


          #create subdirectories
          for folder in structure:
                  folder_path = os.path.join(base_path, folder)
                  os.makedirs(folder_path, exist_ok=True)
                  print(f'Created subdirectory: {folder_path}')

          #Create the README.md file
          readme_path = os.path.join(base_path, 'README.md')
          with open(readme_path,'w') as readme_file:
                  readme_file.write("# Project Overview\n\n")
                  readme_file.write("This project contains the following structure:\n\n")
                  readme_file.write("```\n")
                  readme_file.write(f"{base_path}/\n")
                  for folder in structure:
                              readme_file.write(f"{folder}/\n")
                  readme_file.write("README.md\n")
                  readme_file.write("```\n")
                  readme_file.write("\nInstructions for the project can be added here.")
          print(f"Created file: {readme_path}")

if __name__=='__main__':
        create_project_structure
