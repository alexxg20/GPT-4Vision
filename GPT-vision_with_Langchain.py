import os
import langchain
from langchain.chains import TransformChain
import base64
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain import globals
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from PIL import Image

os.environ["OPENAI_API_KEY"] = ""

# Define image resizing and conversion function
def convert_and_resize(image_path, output_path):
    with Image.open(image_path) as img:
        img.thumbnail((512,512))
        img.save(output_path, "webp")

'''
Before processing any images, we need to load the image data from a file and encode it 
in a format that can be passed to the language model. The function "load_image" takes a dictionary
with an "image_path" key and returns a new dictionary with an "image" key containing the image data
encoded as a base64 string.
This is done using the built-in Python base64 library
'''
def encode_image(output_path):
        with open(output_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
        
def load_image(inputs: dict) -> dict:
    image_path = inputs["image_path"]
    output_path = inputs["output_path"]
    convert_and_resize(image_path, output_path)
    image_base64 = encode_image(output_path)
    return {"image": image_base64}

'''
To integrate this function into a Langchain pipeline, we use TransformChain that takes the "image path"
as input and produces the "image" (a base64-encoded string) as outputCopy code
'''
load_image_chain = TransformChain(
    input_variables=["image_path", "output_path"], 
    output_variables=["image"], 
    transform=load_image
    )

# define the input that will be passed to gpt-vision and the structure of the desired output
class ImageInformation (BaseModel):
    seal_condition: str = Field(description = "intact/partially broken/completely broken")
    surface_condition: str = Field(description = "description of surface condition")
    external_condition: str = Field(description = "description of external condition")
    damage_description: str = Field(description = "detailed damage description")
    potential_damaging_agents: list = Field(description = ["list", "of", "potential", "damaging", "agents"])
    delivered_correctly: str = Field(description = "have the packages been delivered correctly?")
    summary: str = Field(description = "brief summary of the assessment including overall condition and risks")
    #correctly_delivered_packages: int = Field(description = "number of packages delivered without damage")
    #incorrectly_delivered_packages: int = Field(description = "number of packages delivered with damage")
    #surroundings: str = Field(description = "where is the package located?")

# Set verbose
globals.set_debug(True)

# define the chain to interact with the OpenAI API
@chain
def image_model(inputs: dict) -> str | list[str] | dict:
    """Invoke model with image and prompt."""
    model = ChatOpenAI(temperature=0.5, model="gpt-4o", max_tokens=1000)
    msg = model.invoke(
             [HumanMessage(
             content=[
             {"type": "text", "text": inputs["prompt"]},
             {"type": "text", "text": parser.get_format_instructions()},
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}"}},
             ])]
             )
    return msg.content

# define the parser 
parser = JsonOutputParser(pydantic_object=ImageInformation)
# combine everything into a workflow
def get_image_informations(image_path: str, output_path: str) -> dict:
    vision_prompt = """ You are assessing the condition of the package shown in the image after delivery. Complete the following JSON file with valid JSON output:
    """
    vision_chain = load_image_chain | image_model | parser
    result =vision_chain.invoke({'image_path': f'{image_path}','output_path': output_path, 'prompt': vision_prompt})
    print(str(type(result)) + "  here")
    return result


# in my case all images are stored in a similar path, only their number varies.
photo_num = 22
path = f"C:/fotos_gptvision/package_image{photo_num}.jpg" # change to path to your image
output_path = f"C:/fotos_gptvision_resized/package_image{photo_num}_resized.jpg" # change to desired path for resized image

result = get_image_informations(path, output_path)
print(result)
