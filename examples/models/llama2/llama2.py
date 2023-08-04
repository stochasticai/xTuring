# Make the necessary imports
from xturing.models import Llama2

# Load the model
model = Llama2()
# Generate ouputs from the model
outputs = model.generate(texts=["How are you?"])
# Print the generated outputs
print(outputs)

## or

# Make the necessary imports
from xturing.models import BaseModel

# Load the model
model = BaseModel.create("llama2")
# Generate ouputs from the model
outputs = model.generate(texts=["How are you?"])
# Print the generated outputs
print(outputs)
