from zipfile import ZipFile

# Extract the contents of the zip file into the "data" folder
with ZipFile("raw_data.zip", 'r') as zipref:
    zipref.extractall("data")
