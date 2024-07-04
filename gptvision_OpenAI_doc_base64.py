import os
from openai import OpenAI
import base64
from PIL import Image


client = OpenAI(
    api_key="")

# Image resizing and conversion function
def convert_and_resize(image_path, output_path):
    with Image.open(image_path) as img:
        img.thumbnail((512,512))
        img.save(output_path, "webp")

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
photo_num = 5
image_path = f"C:/fotos_gptvision/package_image{photo_num}.jpg"
output_path = f"C:/fotos_gptvision_resized/package_image{photo_num}_resized.jpg" # path where you want to store resized image

# resizing image (to fit low_detail mode)
convert_and_resize(image_path, output_path)

# Getting the base64 string
base64_image = encode_image(output_path)

# calling the gpt model, "content" includes the prompt and the image (in base64)
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {
            "role": "user",
            "content": [
                {"type": "text", "text": """You are assessing the condition of the package shown in the 
                 image after delivery. Complete the following JSON file with valid JSON output:      
                       
    "seal_condition": "intact/partially broken/completely broken",
                 
    "surface_condition": "description of surface condition",
                 
    "external_condition": "description of external condition",  
  
    "damage_description": "detailed damage description",
                       
    "potential_damaging_agents": ["list", "of", "potential", "damaging", "agents"],     
                     
    "summary": "brief summary of the assessment including overall condition and risks"  
                         
  """},
                {
                    "type": "image_url",
                    "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"

                },
                },
            ],
        }
    ],
    max_tokens=300, # specify max number of tokens the output can generate (input is not limited)
)

print(response.choices[0].message.content) # print the content of the response
print(response.usage) # print the amount of tokens used
