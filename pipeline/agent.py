from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

class Agent:
    def __init__(self,model_id,task):
        self.model_id = model_id
        self.task = task

        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

        self.pipeline = pipeline(
            task=task,
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1

        )

        self.agent = HuggingFacePipeline(pipeline=self.pipeline)

    def getModelId(self):
        return self.model_id
    def getTask(self):
        return self.task
    def getAgent(self):
        return self.agent
