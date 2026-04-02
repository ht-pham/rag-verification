from pipeline.agent import Agent
from transformers import pipeline


class Summarizer(Agent):
    def __init__(self,model_id):
        super().__init__(model_id,"summarization")
        
        
        self.pipeline = pipeline(
            task="summarization",
            model=self.model,
            tokenizer=self.tokenizer,
            do_sample=False,
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1

        )

    def summarize(self,chunks):
        
        summarizer = self.getAgent()
        
        first_summary = summarizer(
            chunks,
            max_length=int(len(chunks.split())*0.5),
            max_new_tokens=int(len(chunks.split())*0.5),
            min_length=100,
            do_sample=False)[0]["summary_text"]
        
        final_summary = summarizer(
            first_summary,
            max_length=int(len(first_summary.split())*0.5),
            max_new_tokens=int(len(first_summary.split())*0.5),
            min_length=50,
            do_sample=False
        )[0]["summary_text"]

        return first_summary, final_summary

    def getModelId(self):
        return super().getModelId()
    def getTask(self):
        return super().getTask()
    def getAgent(self):
        return self.pipeline
    
    def clean_text(self,text):
        return " ".join(text.split())