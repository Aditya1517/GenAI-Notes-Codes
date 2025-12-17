from langchain_openai import OpenAI
from dotenv import load_dotenv
from typing import TypedDict
from pydantic import BaseModel, Field

load_dotenv()

model = ChatOpenAI()

class Review(BaseModel):
    
    summary: str = Field(description="A brief summary of the review")
    sentiment:str = Field(description="The overall sentiment of the review, e.g., positive, negative, neutral")
    
structured_model = model.with_structured_output(Review)

result = structured_model.invoke("""The hardware is great, but the software feels bloated. There are too many pre-installed apps that I can't remove. Also, the UI looks outdated compared to other brands. Hoping for a software update to fix this.""")

print(result)
print(result.summary)
print(result.sentiment)