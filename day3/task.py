import os
from pathlib import Path
import pandas as pd
from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, AIMessage

PROVIDER = 'google'  # Change to openai if needed

def secret(name):
    if not os.getenv(name):
        from google.colab import userdata
        os.environ[name] = userdata.get(name)

if PROVIDER == 'google':
    secret('GOOGLE_API_KEY')
    from langchain_google_genai import ChatGoogleGenerativeAI
    model = ChatGoogleGenerativeAI(model=os.getenv('GEMINI_MODEL', 'gemini-2.5-flash'))
elif PROVIDER == 'openai':
    secret('OPENAI_API_KEY')
    from langchain_openai import ChatOpenAI
    model = ChatOpenAI(model=os.getenv('OPENAI_MODEL', 'gpt-4.1-mini'))
else:
    raise ValueError('Unsupported provider')

SYSTEM = '''You are AutoMate AI, an English-only assistant for used-car dealership staff.
Help qualify enquiries, identify missing buyer requirements and draft replies.
Ask about budget, model, body type, fuel, transmission, location, purchase timeline
and contact preference as relevant. Keep customer information private.
No live inventory, CRM, Google Sheets or messaging tools are connected.
Never invent price, stock, mileage, condition, accident history, ownership or loan approval.
Ask staff to verify official records and recommend independent inspection when appropriate.
Label customer messages as drafts requiring staff approval; never claim to send them.
Do not claim to update records or reserve cars. Do not request passwords or OTPs.
Be concise, neutral and professional. Distinguish known facts from unverified details.'''
agent = create_agent(model=model, tools=[], system_prompt=SYSTEM)
history = []

def respond(question, remember=False):
    if not question.strip():
        raise ValueError('Enter a question')
    message = HumanMessage(content=question)
    result = agent.invoke({'messages': (history if remember else []) + [message]})
    answer = result['messages'][-1].content
    if isinstance(answer, list):
        answer = ' '.join(x.get('text','') if isinstance(x, dict) else str(x) for x in answer)
    if remember:
        history.extend([message, AIMessage(content=answer)])
        history[:] = history[-16:]
    return answer

def evaluate(answer, expected):
    groups = [g.split('|') for g in expected.lower().split(';') if g.strip()]
    matches = [any(t.strip() in answer.lower() for t in g if t.strip()) for g in groups]
    return bool(matches) and all(matches), f'{sum(matches)}/{len(matches)}'

def run_tests(path):
    data = pd.read_csv(path)
    if not {'question','expected_groups'}.issubset(data.columns):
        raise ValueError('Invalid CSV columns')
    results = []
    for row in data.itertuples(index=False):
        try:
            answer = respond(row.question)
            passed, score = evaluate(answer, row.expected_groups)
            status = 'PASS' if passed else 'REVIEW'
        except Exception as e:
            answer, score, status = str(e), '-', 'ERROR'
        results.append({'question':row.question,'status':status,'matched':score,'answer':answer})
    report = pd.DataFrame(results)
    print(report.to_string(index=False))
    print(report['status'].value_counts())
    return report

def main():
    print(respond('A customer wants a used sedan. How can I qualify the lead?'))
    print(respond('My name is Anil and my budget is 8 lakh.', True))
    print(respond('What budget did I mention?', True))
    history.clear()
    path = Path(__file__).with_name('test.csv')
    if not path.exists():
        from google.colab import files
        uploaded = files.upload()
        if 'test.csv' not in uploaded:
            raise ValueError('Upload test.csv')
        path = Path('test.csv')
    run_tests(path)
    while True:
        q = input('You: ').strip()
        if q.lower() in ('exit','quit','bye'): break
        if q.lower() == 'reset':
            history.clear()
            print('Memory cleared')
            continue
        try:
            print('AutoMate AI:', respond(q, True))
        except Exception as e:
            print('Agent request failed:', type(e).__name__)

if __name__ == '__main__':
    main()
