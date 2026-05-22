
from pipeline.agent import Agent

class Extractor(Agent):
    def __init__(self,model_id):
        super().__init__(model_id,"text-generation")

    def generate(self,query,context):
        
        prompt = f"""
            You are an information extraction system for biomedical research.

            Your task is ONLY to extract exact sentences from the CONTEXT that are directly relevant to the QUESTION.

            STRICT RULES:
            - Do NOT answer the question.
            - Do NOT summarize.
            - Do NOT explain.
            - ONLY copy exact sentences from the CONTEXT.
            - Each sentence must include its chunk ID (e.g. [1], [2]).
            - If no relevant sentence exists, return exactly: None

            CONTEXT
            {context}

            QUESTION:
            {query}

            OUTPUT FORMAT
            - [id] exact sentence
            - [id] exact sentence

            If nothing is relevant: 
            None
            """
        extractor = self.get_agent()
        self.answer = extractor.invoke([prompt])
        return self.answer
    
    def printOutput(self):
        print(self.answer)

    def get_model_id(self):
        return super().get_model_id()
    def get_task(self):
        return super().get_task()
    def get_agent(self):
        return super().get_agent()