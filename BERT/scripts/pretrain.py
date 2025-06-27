import torch
import torch.nn as nn
import pickle
from torch.utils.data import DataLoader, Dataset
from torch.optim import AdamW
import pickle
import tqdm
from pathlib import Path
import sys

sys.path.append(str(Path(__file__).resolve().parent.parent))

from my_bert.model import BertForPreTraining

class BERTDataset(Dataset):
    def __init__(self, data_path):
        super().__init__()
        with open(data_path, 'rb') as file:
            self.examples = pickle.load(file)
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, index):
        return self.examples[index]
    
VOCAB_SIZE = 10000
MAX_SEQ_LEN = 128

D_MODEL = 256
NUM_LAYERS = 4
NUM_HEADS = 8
D_FF = D_MODEL * 4
DROPOUT_PROB = 0.1

BATCH_SIZE = 64
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
SAVE_EVERY_N_EPOCHS = 2

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    project_root = Path(__file__).parent.parent
    processed_data_path = project_root / 'data' / 'processed' / 'bert_data.pkl'
    model_save_dir = project_root / 'models'
    model_save_dir.mkdir(parents=True, exist_ok=True)

    print('Loading pre-processed data: ')
    dataset = BERTDataset(processed_data_path)
    train_loader = DataLoader(dataset, batch_size = BATCH_SIZE, shuffle = True)
    print(f'    ...found {len(dataset)} training examples.')

    print('Initializing BERT model: ')
    model = BertForPreTraining(
        vocab_size = VOCAB_SIZE,
        d_model = D_MODEL,
        num_layers = NUM_LAYERS,
        d_ff = D_FF,
        max_len = MAX_SEQ_LEN,
        dropout_prob = DROPOUT_PROB,
    ).to(device)

    criterion = nn.CrossEntropyLoss(ignore_index = -100)
    optimizer = torch.optim.AdamW(model.parameters(), lr = LEARNING_RATE)

    print('Starting training: ')
    for epoch in range(NUM_EPOCHS):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc = f'Epoch {epoch + 1}/{NUM_EPOCHS}')

        for batch in progress_bar:
            input_ids = batch['input_ids'].to(device)
            segment_ids = batch['segment_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            mlm_labels = batch['mlm_labels'].to(device)
            nsp_label = batch['nsp_label'].to(device)

            mlm_logits, nsp_logits = model(input_ids, segment_ids, attention_mask)

            mlm_loss = criterion(mlm_logits.view(-1, VOCAB_SIZE), mlm_labels.view(-1))
            
            nsp_loss = criterion(nsp_logits, nsp_label)

            loss = mlm_loss + nsp_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix(loss = loss.item())
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch {epoch + 1}/{NUM_EPOCHS} | Average Loss: {avg_loss:.4f}')

        if (epoch + 1) % SAVE_EVERY_N_EPOCHS == 0:
            checkpoint_path = model_save_dir / f'bert_pretrained_epoch_{epoch + 1}.pth'
            print(f'Saving model checkpoint to {checkpoint_path}...')
            torch.save(model.state_dict(), checkpoint_path)

    print('\nTraining complete')
    final_model_path = model_save_dir / 'bert_pretrained_final.pth'
    print(f'Saving final model to {final_model_path}')
    torch.save(model.state_dict(), final_model_path)

if __name__ == '__main__':
    main()