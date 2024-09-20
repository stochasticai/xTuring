# Make the necessary imports
from xturing.models import Mistral

# Load the model
model = Mistral()
# Generate ouputs from the model
outputs = model.generate(texts=["How are you?"])
# Print the generated outputs
print(outputs)

## or

# Make the necessary imports
from xturing.models import BaseModel

# Load the model
model = BaseModel.create("mistral")
# Generate ouputs from the model
outputs = model.generate(texts=["How are you?"])
# Print the generated outputs
print(outputs)
