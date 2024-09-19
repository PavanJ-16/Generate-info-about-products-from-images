# pip install ollama
# ollama run modelname 
# replace modelname with desired visual language model



import os
import pandas as pd
import requests
import ollama

def download_image(url, filename):
    
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(filename, 'wb') as f:
        for chunk in response.iter_content(chunk_size=1024):
            if chunk:
                f.write(chunk)
        print(f"Image downloaded successfully: {filename}")



def predictor(image_link, category_id, entity_name):
   
    url = image_link  
    filename = "downloaded_image"
    download_image(url, filename)
    res = ollama.chat(
	model="0ssamaak0/xtuner-llava:phi3-mini-int4", #Can be changed with visual language models supported by ollama
	messages=[
		{
			'role': 'user',
			'content': f'return the entity {entity_name} info about the product in the image in single word',
			'images': ['./downloaded_image'],
            'role': 'system',
			'content': 'return info in single word about only entity quanitity that is asked. Use the following mapping for units of measurement: entity_unit_map = width: centimetre, foot, millimetre, metre, inch, yard; depth: centimetre, foot, millimetre, metre, inch, yard; height: centimetre, foot, millimetre, metre, inch, yard; item_weight: milligram, kilogram, microgram, gram, ounce, ton, pound; maximum_weight_recommendation: milligram, kilogram, microgram, gram, ounce, ton, pound; voltage: millivolt, kilovolt, volt; wattage: kilowatt, watt; item_volume: cubic foot, microlitre, cup, fluid ounce, centilitre, imperial gallon, pint, decilitre, litre, millilitre, quart, cubic inch, gallon. Use only the units/valuse asked. If asked entity is not in image, return null',
		}
	]
)
    
    return res['message']['content']



if __name__ == "__main__":
    DATASET_FOLDER = '../dataset/'
    
    test = pd.read_csv(os.path.join(DATASET_FOLDER, 'test.csv'))
    
    test['prediction'] = test.apply(
        lambda row: predictor(row['image_link'], row['group_id'], row['entity_name']), axis=1)
    
    output_filename = os.path.join(DATASET_FOLDER, 'test_out.csv')
    test[['index', 'prediction']].to_csv(output_filename, index=False)