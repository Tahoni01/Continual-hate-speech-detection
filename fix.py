import nbformat

path = "Main.ipynb"

nb = nbformat.read(path, as_version=4)

if "widgets" in nb["metadata"]:
    del nb["metadata"]["widgets"]

nbformat.write(nb, path)

print("Notebook sistemato")