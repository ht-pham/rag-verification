from pipeline.agent import Agent

class Summarizer(Agent):
    def __init__(self, model_id):
        super().__init__(model_id, "text-generation")

    def summarize(self, chunks):
        
        chunks = self.clean_text(chunks)

        model = self.model
        #prompt = f" {chunks}\n\n"
        prompt = f"Summarize the following context:\n\n{chunks}\n\nSummary:"

        # Tokenize inputs as PyTorch tensors
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(model.device)

        # Return LongTensor [batch_size,max_length OR max_new_tokens]
        outputs = model.generate(
            **inputs, 
            max_length=int(len(chunks.split()) * 0.5),
            max_new_tokens=int(len(chunks.split()) * 0.5)
        )
        output_tokens = outputs[0,inputs.input_ids.shape[-1]:]

        summary = self.tokenizer.decode(
            output_tokens, 
            skip_special_tokens=True
        )
        
        return summary

    def getModelId(self):
        return super().getModelId()
    def getTask(self):
        return super().getTask()
    def getAgent(self):
        return self.model
    
    def clean_text(self,text):
        return " ".join(text.split())