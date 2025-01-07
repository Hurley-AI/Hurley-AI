from hurley_agentic.agent import Agent
from hurley_agentic.agent_endpoint import start_app

from dotenv import load_dotenv
load_dotenv(override=True)

customer_id = '1366999410'
corpus_id = '1'
api_key = 'zqt_UXrBcnI2UXINZkrv4g1tQPhzj02vfdtqYJIDiA'

assistant = Agent.from_corpus(
    tool_name = 'query_hurley_website',
    hurley_customer_id = customer_id,
    hurley_corpus_id = corpus_id,
    hurley_api_key = api_key,
    data_description = 'Data from hurley.com website',
    assistant_specialty = 'hurley'
)

start_app(assistant, host="0.0.0.0", port=8000)
