from langchain.agents import create_agent
from langsmith.env import get_system_metrics

from agent.tools.agent_tools import get_weather, get_user_location, get_current_month, \
    get_user_id, fetch_external_data, fill_context_for_report
from agent.tools.middleware import monitor_tool, log_before_model, report_prompt_switch
from model.factory import chat_model
from utils.prompt_load import load_system_prompt


class ReactAgent():
    def __init__(self):
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompt(),
            tools=[get_system_metrics, get_weather, get_user_location, get_user_id, get_current_month,
                   fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )

    def execute_stream(self, query: str):
        input_dict = {"message": [{"role": "user", "content": query}]}
        for chuck in self.agent.stream(input_dict, stream_mode="values", context={"report": False}):
            latest_message = chuck["message"][-1]
            if latest_message.content:
                yield latest_message.content.strip()


if __name__ == '__main__':
    agent = ReactAgent()
    for chuck in agent.execute_stream("扫地机器人在我所在的地区的气温下如何保养"):
        print(chuck, end="", flush=True)
