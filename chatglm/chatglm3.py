import ast
import json

from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Mapping, Optional, Any
from langchain.llms.base import LLM
from langchain_core.output_parsers.transform import BaseTransformOutputParser


class ResponseParser(BaseTransformOutputParser[str]):
    """OutputParser that parses LLMResult into the top likely string."""

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True

    @classmethod
    def get_lc_namespace(cls) -> List[str]:
        """Get the namespace of the langchain object."""
        return ["langchain", "schema", "output_parser"]

    @property
    def _type(self) -> str:
        """Return the output parser type for serialization."""
        return "default"

    def parse(self, text: str) -> str:
        '''
Action: 
```
{"action": "Final Answer", "action_input": "666"}
```'''
        text = text.strip()
        text = text.lstrip("\nAction: \n```\n").rstrip("\n```")
        obj = json.loads(text)
        return obj.get("action_input")


class ChatGLM3(LLM):
    tokenizer: object = None
    model: object = None
    history: List = []
    has_search: bool = False
    max_token: int = 8192
    do_sample: bool = True
    temperature: float = 0.8
    top_k = 50
    top_p = 0.8

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)

    @property
    def _llm_type(self) -> str:
        return "ChatGLM3"
    
    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        """Get the identifying parameters."""
        return {
            "max_token": self.max_token,
            "do_sample": self.do_sample,
            "temperature": self.temperature,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

    def load_model(self, model_name_or_path=None):
        print(f"[ChatGLM3] load from {model_name_or_path}")
        model_config = AutoConfig.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )
        self.model = AutoModel.from_pretrained(
            model_name_or_path, config=model_config, trust_remote_code=True, device_map="auto").eval()

    def _tool_history(self, prompt: str):
        ans = []

        tools_prompt_list = prompt.split(
            "You have access to the following tools:\n\n")
        
        if len(tools_prompt_list) < 2:
            return ans, prompt

        tool_prompts = tools_prompt_list[1].split("\n\nUse a json blob")[0].split("\n")
        tools_json = []

        for tool_desc in tool_prompts:
            name = tool_desc.split(":")[0]
            description = tool_desc.split(", args:")[0].split(":")[1].strip()
            parameters_str = tool_desc.split("args:")[1].strip()
            parameters_dict = ast.literal_eval(parameters_str)
            params_cleaned = {}
            for param, details in parameters_dict.items():
                params_cleaned[param] = {'description': details['description'], 'type': details['type']}

            tools_json.append({
                "name": name,
                "description": description,
                "parameters": params_cleaned
            })

        ans.append({
            "role": "system",
            "content": "Answer the following questions as best as you can. You have access to the following tools:",
            "tools": tools_json
        })

        dialog_parts = prompt.split("Human: ")
        for part in dialog_parts[1:]:
            if "\nAI: " in part:
                user_input, ai_response = part.split("\nAI: ")
                ai_response = ai_response.split("\n")[0]
            else:
                user_input = part
                ai_response = None

            ans.append({"role": "user", "content": user_input.strip()})
            if ai_response:
                ans.append({"role": "assistant", "content": ai_response.strip()})

        query = dialog_parts[-1].split("\n")[0]
        return ans, query

    def _extract_observation(self, prompt: str):
        return_json = prompt.split("Observation: ")[-1].split("\nThought:")[0]
        self.history.append({
            "role": "observation",
            "content": return_json
        })
        return

    def _extract_tool(self):
        if len(self.history[-1]["metadata"]) > 0:
            metadata = self.history[-1]["metadata"]
            content = self.history[-1]["content"]
            # print(f"content: {content}")

            lines = content.split('\n')
            for line in lines:
                if 'tool_call(' in line and ')' in line and self.has_search is False:
                    # 获取括号内的字符串
                    params_str = line.split('tool_call(')[-1].split(')')[0]

                    # 解析参数对
                    params_pairs = [param.split("=") for param in params_str.split(",") if "=" in param]
                    params = {pair[0].strip(): pair[1].strip().strip("'\"") for pair in params_pairs}
                    action_json = {
                        "action": metadata,
                        "action_input": params
                    }
                    self.has_search = True
                    # print("*****Action*****")
                    # print(action_json)
                    # print("*****Answer*****")
                    return f"""
Action: 
```
{json.dumps(action_json, ensure_ascii=False)}
```"""
        final_answer_json = {
            "action": "Final Answer",
            "action_input": self.history[-1]["content"]
        }
        self.has_search = False
        return f"""
Action: 
```
{json.dumps(final_answer_json, ensure_ascii=False)}
```"""

    def _call(
        self,
        prompt: str,
        history: List = [],
        stop: Optional[List[str]] = ["<|user|>"],
        ):
        if not self.has_search:
            self.history, query = self._tool_history(prompt)
        else:
            self._extract_observation(prompt)
            query = ""
        # print(f"\n1: history: {json.dumps(self.history, ensure_ascii=False)}\n")
        # print(f"query: {query}\n")
        print(f"temperature: {self.temperature} max_token: {self.max_token}",
            f"top_k: {self.top_k} top_p: {self.top_p}")
        _, self.history = self.model.chat(
            self.tokenizer,
            query,
            history=self.history,
            do_sample=self.do_sample,
            max_length=self.max_token,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
        )
        # print(f"\n2: history: {json.dumps(self.history, ensure_ascii=False)}\n")
        response = self._extract_tool()
        # print(f"\n2: response: {json.dumps(response, ensure_ascii=False)}\n")
        history.append((prompt, response))
        return response
