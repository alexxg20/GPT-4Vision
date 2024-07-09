import os
import langchain
from langchain.chains import TransformChain
import base64
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain import globals
from langchain_core.runnables import chain
from langchain_core.output_parsers import JsonOutputParser
from PIL import Image

os.environ["OPENAI_API_KEY"] = "your_api_here"

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
This is done using the Python base64 library
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
    damage_seal: str = Field(description = "Is there any damage in the package's seal? Yes/No")
    damage_surface: str = Field(description = "Is there any damage in the package's surface? Yes/No")
    external_condition: str = Field(description = "Is there any external damage? Yes/No")
    damage_description: str = Field(description = "Is the package severely damaged? Yes/No")
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
    model = ChatOpenAI(temperature=0.0, model="gpt-4o", max_tokens=300)
    msg = model.invoke(
             [
             # to give some more context to the model, you can specify their role 
             # or give any additional context using SystemMessage
             SystemMessage(content="You are an AI bot that helps the user assess the state of a package."
                            ),
             HumanMessage(
             content=[
             {"type": "text", "text": inputs["prompt"]},
             {"type": "text", "text": parser.get_format_instructions()},
             {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{inputs['image']}"}},
             ])#,

             # If the model has already responded, or you want to simulate it, use AIMessage, 
             # then HumanMessage again to engage in a conversation with the model
            
             #AIMessage(content = ""),
            
             #HumanMessage(content = "Then, would you say the delivery has been successful?")
             
             ])
    return msg.content

# define the parser 
parser = JsonOutputParser(pydantic_object=ImageInformation)
# combine everything into a workflow
def get_image_informations(image_path: str, output_path: str) -> dict:
    vision_prompt = """ You are assessing the condition of the package shown in the image after delivery. Complete the following JSON file with valid JSON output:
    """
    vision_chain = load_image_chain | image_model | parser
    result =vision_chain.invoke({'image_path': f'{image_path}','output_path': output_path, 'prompt': vision_prompt})
    return result


# all images are stored in a similar path, only the number varies.
photo_num = 16
path = f"C:/fotos_gptvision/package_image{photo_num}.jpg" # path to our image
output_path = f"C:/fotos_gptvision_resized/package_image{photo_num}_resized.jpg"
result = get_image_informations(path, output_path)
print(result)
