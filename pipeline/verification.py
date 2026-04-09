from pipeline.agent import Agent
from transformers import AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

class Verifier(Agent):
    def __init__(self,model_id):
        super().__init__(model_id,"zero-shot-classification")
        self.agent = self.pipeline
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)

    def classify(self,query,evidence):

        hypothesis = f"{query}"
        premises = evidence

        # Manually tokenize and prepare inputs for the model
        inputs = self.tokenizer(
            premises,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        inputs = {k: v.to(self.device) for k,v in inputs.items()}
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Apply softmax to convert to probabilities of outputs
        logits = outputs.logits
        probs = F.softmax(logits,dim=1)

        # 0 = contradiction, 1 = neutral, 2 = entailment
        contradiction, neutral, entailment = probs[0]
        label = torch.argmax(probs).item()
        print("*** Verifying the answer with the retrieved evidence: ***")
        print(f"Label: {label}")
        print(f"Contradiction: {contradiction:.4f}, Neutral: {neutral:.4f}, Entailment: {entailment:.4f}")
        if label == 2:
            return "Yes", entailment.item()
        elif label == 0:
            return "No", contradiction.item()
        else:
            return "Not enough information", neutral.item()

        
    
    def getModelId(self):
        return super().getModelId()
    def getTask(self):
        return super().getTask()
    def getAgent(self):
        return self.pipeline