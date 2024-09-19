# Generate-info-about-products-from-images

Developed this for the Amazon ML challenge.

It accepts a CSV, with following fields: index, image_link, group_id, entity_name, entity_value

The visual model used, processes the image and generates reponses.

An output file is a CSV with data fields index and prediction about the entity asked.



# Needs Ollama and the desired model installed to run.

pip install ollama

ollama run modelname 

replace modelname with desired visual language model

