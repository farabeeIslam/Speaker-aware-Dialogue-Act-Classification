import torch
import torch.nn as nn
import os
import numpy as np
from torch.optim import AdamW
from models_t5_swda import T5DialogueClassifierSWDA
from sklearn.metrics import accuracy_score

class EngineT5SWDA:
    def __init__(self, args, tokenizer):
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
        self.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        os.makedirs('ckp', exist_ok=True)

        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        np.random.seed(args.seed)

        from datasets import data_loader_t5
        self.train_loader = data_loader_t5(args, 'train')
        self.val_loader = data_loader_t5(args, 'val')
        self.test_loader = data_loader_t5(args, 'test')

        print('Initializing T5 model for SWDA...')
        self.model = T5DialogueClassifierSWDA().to(self.device)
        self.tokenizer = tokenizer
        self.optimizer = AdamW(self.model.parameters(), lr=args.lr)
        self.args = args

    def train(self):
        best_val_acc = 0
        for epoch in range(self.args.epochs):
            print(f"Epoch {epoch+1}")
            self.model.train()
            for batch in self.train_loader:
                texts = batch['text']
                labels = batch['label']
                labels = [str(label) for label in labels]

                enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
                targets = self.tokenizer(labels, padding=True, truncation=True, return_tensors='pt').to(self.device)

                self.optimizer.zero_grad()
                loss, _ = self.model(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    labels=targets['input_ids']
                )
                loss.backward()
                self.optimizer.step()

            acc = self.evaluate(self.val_loader)
            print(f"Validation Accuracy: {acc:.3f}")
            if acc > best_val_acc:
                best_val_acc = acc
                torch.save(self.model.state_dict(), f"ckp/model_t5_swda.pt")
        print("Training complete.")
        self.model.load_state_dict(torch.load(f"ckp/model_t5_swda.pt"))
        test_acc = self.evaluate(self.test_loader)
        print(f"Test Accuracy: {test_acc:.3f}")

    def evaluate(self, loader):
        self.model.eval()
        y_true = []
        y_pred = []
        with torch.no_grad():
            for batch in loader:
                texts = [str(t) for t in batch['text']]
                labels = [str(l) for l in batch['label']]

                enc = self.tokenizer(texts, padding=True, truncation=True, return_tensors='pt').to(self.device)
                outputs = self.model.model.generate(
                    input_ids=enc['input_ids'],
                    attention_mask=enc['attention_mask'],
                    max_new_tokens=10  # avoids length warning
                )
                preds = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

                y_pred.extend(preds)
                y_true.extend(labels)

        correct = sum([p == t for p, t in zip(y_pred, y_true)])
        return correct / len(y_true)



