import os
import json
from functools import wraps
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.tools import tool

if not os.getenv('GROQ_API_KEY'):
    from google.colab import userdata
    os.environ['GROQ_API_KEY'] = userdata.get('GROQ_API_KEY')
llm = ChatGroq(model='openai/gpt-oss-120b', temperature=0)

SYSTEM = '''You are AutoMate AI, an English-only automotive assistant.
Explain vehicle choices, maintenance and safety in simple language.
Ask about budget and usage when needed. You have no live prices or inventory.
Do not invent specifications, mileage or availability. Recommend official
verification and independent inspection for used cars. Do not claim to
contact dealers or provide professional mechanical or financial advice.'''

prompt = ChatPromptTemplate.from_messages([
    ('system', '''Classify the request using exactly one label.
Affordable car for a small family -> recommendation
Diesel versus petrol -> comparison
Engine overheating -> maintenance
How airbags work -> safety
How to sell a car -> other'''),
    ('human', '{question}')])
classifier = prompt | llm | StrOutputParser()

def classify(question):
    label = classifier.invoke({'question': question}).strip().lower()
    return label if label in ['recommendation','comparison','maintenance','safety','other'] else 'other'

extract_prompt = ChatPromptTemplate.from_messages([
    ('system', '''Return only a JSON object with keys intent, budget_inr,
vehicle_type, fuel_preference, use_case, missing_information.
Use the classification labels recommendation, comparison, maintenance,
safety or other. Convert budget to rupees. Use null for unknown values
and a list of strings for missing_information. Do not guess.'''),
    ('human', '{question}')])
extract_chain = extract_prompt | llm | StrOutputParser()

def extract_enquiry(question):
    data = json.loads(extract_chain.invoke({'question': question}))
    keys = {'intent','budget_inr','vehicle_type','fuel_preference','use_case','missing_information'}
    if not isinstance(data, dict) or set(data) != keys:
        raise ValueError('Unexpected JSON structure')
    return data

categories = {'hatchback':'Compact body, useful for urban driving.',
              'sedan':'Separate luggage compartment and passenger cabin.',
              'suv':'Typically taller body and higher seating position.'}

def estimate_total(price, quantity=1):
    """Calculate total cost using a user-supplied price."""
    if price < 0 or quantity < 0:
        raise ValueError('Values must be non-negative')
    return price * quantity

def log_call(fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        print('Running:', fn.__name__)
        return fn(*args, **kwargs)
    return wrapper

@log_call
def category_note(vehicle_type):
    return categories.get(vehicle_type.strip().lower(), 'Category not found.')

@tool
def vehicle_guide(vehicle_type: str) -> str:
    """Explain a general vehicle body category."""
    return category_note(vehicle_type)

history = []
chat_prompt = ChatPromptTemplate.from_messages([
    ('system', SYSTEM), MessagesPlaceholder('history'), ('human', '{question}')])
chat = chat_prompt | llm | StrOutputParser()

def ask(question):
    if not question.strip():
        raise ValueError('Enter a question')
    answer = chat.invoke({'history': history, 'question': question})
    history.extend([HumanMessage(content=question), AIMessage(content=answer)])
    history[:] = history[-16:]
    return answer

def main():
    print(classify('How does electronic stability control work?'))
    print(json.dumps(extract_enquiry('Need a hybrid sedan under 15 lakh for commuting'), indent=2))
    print(estimate_total(1200, 4), estimate_total.__doc__)
    print(category_note('unknown'))
    print(vehicle_guide.name, vehicle_guide.description)
    print(vehicle_guide.invoke({'vehicle_type':'sedan'}))
    while True:
        q = input('You: ').strip()
        if q.lower() in ('exit','quit','bye'): break
        if q.lower() == 'reset':
            history.clear()
            print('Memory cleared')
            continue
        try:
            print('AutoMate AI:', ask(q))
        except Exception as e:
            print('Request failed:', type(e).__name__)

if __name__ == '__main__':
    main()
