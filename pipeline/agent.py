from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, AutoModelForSequenceClassification, AutoModelForCausalLM
from langchain_huggingface import HuggingFacePipeline
from transformers import pipeline

class Agent:
    def __init__(self, model_id, task):
        self.model_id = model_id
        self.task = task
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)

        if task == "zero-shot-classification":
            self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
            
        else:
            self.model = AutoModelForCausalLM.from_pretrained(model_id,device_map="auto",dtype="auto")
        
    
        self.pipeline = pipeline(task=task, model=self.model, tokenizer=self.tokenizer)
        self.agent = self.pipeline

    def getModelId(self):
        return self.model_id

    def getTask(self):
        return self.task

    def getAgent(self):
        return self.agent
