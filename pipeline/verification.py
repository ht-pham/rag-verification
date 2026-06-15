from pipeline.agent import Agent
import torch
import torch.nn.functional as F

class Verifier(Agent):
    def __init__(self, model_id):
        super().__init__(model_id, "zero-shot-classification")
        self.agent = self.pipeline
        self.device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        self.model = self.model.to(self.device)

    def classify(self, query, evidence):
        hypothesis = f"{query}"
        premises = evidence

        # Manually tokenize and prepare input tensors for the model
        inputs = self.tokenizer(
            premises,
            hypothesis,
            return_tensors="pt",
            truncation=True,
            padding=True
        )
        # Move inputs to the same device as the model
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Disable gradient calculations for inference
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
        print(f"Contradiction [0]: {contradiction:.4f}, Neutral [1]: {neutral:.4f}, Entailment [2]: {entailment:.4f}")
        if label == 2:
            return "Yes", entailment.item()
        elif label == 0:
            return "No", contradiction.item()
        else:
            return "Not enough information", neutral.item()

    def vote(self,votes,alpha):
        # Hybrid score approach: H = alpha*Max(Fraction of votes) + (1-alpha)*Max(Approval Score)
        # Total number of votes = total number of relevant-evaluated retrieved documents
        total = len(votes)
        
        # Majority Votes = Statistics of labels
        # Getting stats
        contradicts = sum(1 for label,score in votes if label == "No")
        neutrals = sum(1 for label,score in votes if label == "Not enough information")
        entailments = sum(1 for label,score in votes if label == "Yes")
        stats = {-1:contradicts,0:neutrals,1:entailments}

        # Compute fractions for each class
        votes_fraction = {-1:contradicts/total,0:neutrals/total,1:entailments/total}
        most_voted = max(votes_fraction.values())
        majority_vote = [k for k,v in votes_fraction.items() if v == most_voted] # there is a simpler way, but this way ensure a tie is not missed

        # Scoring Votes = Approval score based on probability scores 
        no = sum(score for label,score in votes if label == "No")/total
        neutral = sum(score for label,score in votes if label == "Not enough information")/total
        yes = sum(score for label,score in votes if label == "Yes")/total
        scores = {-1:no,0:neutral,1:yes}
        highest_approved_score = max(scores.values())
        approved_class = [k for k,v in scores.items() if v == highest_approved_score]

        # Case 1: Both Majority Vote and Scoring Vote agree on one single label
        consensus = majority_vote == approved_class
        if consensus == True and len(majority_vote) == 1:
            print(f"Consensus: True")
            print(f"Majority Vote: {majority_vote[0]} with {stats[majority_vote[0]]} out of {total} retrieved documents as evidence")
            print(f"Approval score: {highest_approved_score:.4f}%")

            # return final label with its fraction of votes and its approval score
            return majority_vote[0], most_voted, highest_approved_score
        # Case 2: Both Majority Vote and Scoring Vote agree on two labels (i.e. there is a tie)
        elif consensus == True and len(majority_vote) != 1:
            ties = {}
            for label in majority_vote:
                ties[label] = alpha * votes_fraction[label] + (1-alpha)*scores[label]
            final_label = [k for k,v in ties.items() if v == max(ties.values())]
            
            print(f"Consensus: True but there is a tie")
            print(f"Final aggregated vote: {final_label[0]} with {stats[majority_vote[0]]} out of {total} retrieved documents as evidence")
            print(f"Approval score: {ties[final_label[0]]}%")

            # return final label with its fraction of votes and its approval score
            return final_label[0], votes_fraction[final_label[0]] , ties[final_label[0]]
        
        # Case 3: Majority Vote and Scoring Vote disagree on labels
        else:
            hybrid_voting = {-1:0.0,0:0.0,1:0.0}
            for label in hybrid_voting.keys():
                hybrid_voting[label] = alpha * votes_fraction[label] + (1-alpha)*scores[label]
            final_label = [k for k,v in hybrid_voting.items() if v == max(hybrid_voting.values())]
            print(f"Consensus: True but there is a tie")
            print(f"Final aggregated vote: {final_label[0]} with {stats[majority_vote[0]]} out of {total} retrieved documents as evidence")
            print(f"Approval score: {hybrid_voting[final_label[0]]}%")

            return final_label[0], votes_fraction[final_label[0]] , hybrid_voting[final_label[0]]

    def get_model_id(self):
        return super().get_model_id()
    def get_task(self):
        return super().get_task()
    def get_agent(self):
        return self.pipeline