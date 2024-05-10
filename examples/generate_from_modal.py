import logfire
logfire.install_auto_tracing(modules=['cynde'])
logfire.configure(pydantic_plugin=logfire.PydanticPlugin(record='all'))
from typing import List
from pydantic import BaseModel
import polars as pl
from cynde.functional.generate.generate_modal import generate_column,validate_df
from cynde.functional.generate.types import InstructionConfig

class Animals(BaseModel):
    location: str
    activity: str
    animals_seen: int  # Constrained integer type
    animals: List[str]

system = "Respond with a valid json parsing the following sentence: \n"
# texts = ["I saw a puppy a cat and a raccoon during my bike ride in the park","I saw a  crocodile and a snake in the river","DUring camping I saw a bear and a deer"]
texts = [
    "While hiking, I spotted a squirrel, a fox, and a rabbit in the forest.",
    "I noticed a dolphin gang and a jellyfish swimming in the ocean.",
    "During a safari, I observed an elephant and a lion roaming across the savannah."
]
# Additional similar sentences
texts = [
    "On my nature walk, I glimpsed a butterfly, a hedgehog, and a sparrow among the bushes.",
    "At the aquarium, I marveled at a seahorse, a manta ray, and a clownfish in the reef exhibit.",
    "In the savannah, I witnessed a cheetah chasing a gazelle across the grasslands.",
    "During my evening stroll, I observed a possum, a skunk, and a barn owl along the trail.",
    "On the farm, I encountered a rooster, a sheep, and a friendly goat near the barn.",
    "In the wetland, I saw a heron, a frog, and a muskrat in the water.",
    "While snorkeling, I spotted a parrotfish, a sea turtle, and a moray eel near the coral reef.",
    "On the safari, I caught a glimpse of a rhinoceros and a leopard resting under the acacia trees.",
    "In the meadow, I noticed a hare, a woodpecker, and a bumblebee buzzing around the flowers.",
    "During the morning jog, I came across a beaver, a red fox, and a mallard duck by the stream.",
    "At the wildlife sanctuary, I watched a condor and a jackal circling over the rocky outcrops.",
    "In the jungle, I saw a toucan, a capybara, and a jaguar hiding in the canopy.",
    "At the riverbank, I noticed a kingfisher, a salamander, and a dragonfly skimming the water.",
    "During a forest trek, I saw a lynx, a pine marten, and a wild boar rustling through the underbrush.",
    "In the desert, I stumbled upon a fennec fox and a scorpion seeking shade under a cactus."
]

df = pl.DataFrame(data={"text":texts})
instruction = InstructionConfig(system_prompt=system,column="text",output_schema=Animals.model_json_schema(),modal_endpoint="example-tgi-Meta-Llama-3-8B-Instruct")
out_df = generate_column(df,instruction)
validated_df = validate_df(out_df,Animals)

print(validated_df)
for generation,error in zip(validated_df["text_Animals"],validated_df["text_Animals_errors"]):
    print(generation)
    print(error)