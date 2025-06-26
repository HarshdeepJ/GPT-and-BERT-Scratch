import re
import random
import pickle
from pathlib import Path
from collections import Counter

import torch
import nltk
from tqdm import tqdm

MAX_SEQ_LEN = 128
VOCAB_SIZE = 10000
MASK_PROB = 0.15

def clean_text(text):
    text = re.sub(r'[A-Z\s]+:', '', text, flags = re.MULTILINE)
    text = re.sub(r'^[A-Z\s]+$', '', text, flags = re.MULTILINE)
    text = ' '.join(text.replace('\n', ' ').split())
    text = text.lower()
    return text

class Tokenizer:
    def __init__(self, word_to_idx, vocab_size):
        self.word_to_idx = word_to_idx
        self.idx_to_word = {idx: word for word, idx in word_to_idx.items()}
        self.vocab_size = vocab_size
    
    def tokenize(self, text):
        return re.findall(r"[\w']+|.,!?;]", text)
    
    def convert_tokens_to_ids(self, tokens):
        unk_id = self.word_to_idx.get('[UNK]')
        return [self.word_to_idx.get(token, unk_id) for token in tokens]

    @classmethod
    def build(cls, sentences, vocab_size):
        word_counts = Counter()
        for sentence in sentences:
            word_counts.update(re.findall(r"[\w']+|[.,!?;]", sentence))
        special_token = ['[PAD]', '[UNK]', '[CLS]', '[SEP]', '[MASK]']
        word_to_idx = {token: i for i, token in enumerate(special_token)}

        most_common_words = word_counts.most_common(vocab_size - len(special_token))
        for word, _ in most_common_words:
            if word not in word_to_idx:
                word_to_idx[word] = len(word_to_idx)
        
        return cls(word_to_idx, vocab_size)

def create_sentence_pairs(sentences):
    pairs = []
    num_sentences = len(sentences)
    for i in range(num_sentences -1):
        sent_a = sentences[i]

        if random.random() < 0.5:
            sent_b = sentences[i+1]
            is_next = 0
        
        else:
            random_idx = random.randint(0, num_sentences - 1)
            while random_idx == i or random_idx == i+1:
                random_idx = random.randint(0, num_sentences - 1)
            sent_b = sentences[random_idx]
            is_next = 1
        pairs.append((sent_a, sent_b, is_next))
    return pairs

def _truncate_seq_pair(tokens_a, tokens_b, max_num_tokens):
    while True:
        total_length = len(tokens_a) + len(tokens_b)
        if total_length <= max_num_tokens:
            break

        if len(tokens_a) > len(tokens_b):
            tokens_a.pop()
        else:
            tokens_b.pop()

def create_training_example(pair, tokenizer, max_len, mask_prob):
    sent_a, sent_b, nsp_label = pair
    tokens_a = tokenizer.tokenize(sent_a)
    tokens_b = tokenizer.tokenize(sent_b)

    _truncate_seq_pair(tokens_a, tokens_b, max_len - 3)
    tokens = ['[CLS]'] + tokens_a + ['[SEP]'] + tokens_b + ['[SEP]']

    segment_ids = [0]*(len(tokens_a) + 2) + [1]*(len(tokens_b) + 1)

    mlm_input_tokens = list(tokens)
    mlm_labels = [-100] * max_len

    for i, token in enumerate(tokens):
        if token in ['[CLS]', '[SEP]']:
            continue

        if random.random() < mask_prob:
            original_token_id = tokenizer.convert_tokens_to_ids([token])[0]
            mlm_labels[i] = original_token_id

            if random.random() < 0.8:
                mlm_input_tokens[i] = '[MASK]'
            elif random.random() < 0.5:
                num_special_tokens = 5
                random_word_id = random.randint(num_special_tokens, tokenizer.vocab_size - 1)
                mlm_input_tokens[i] = tokenizer.idx_to_word[random_word_id]
        
        input_ids = tokenizer.convert_tokens_to_ids(mlm_input_tokens)
        padding_len = max_len - len(input_ids)
        input_ids.extend([0] * padding_len)
        segment_ids.extend([0] * padding_len)

        attention_mask = [1] * (len(tokens)) + [0] * padding_len

    return {
        'input_ids': torch.tensor(input_ids, dtype = torch.long),
        'segment_ids': torch.tensor(segment_ids, dtype = torch.long),
        'attention_mask': torch.tensor(attention_mask, dtype = torch.long),
        'mlm_labels': torch.tensor(mlm_labels, dtype = torch.long),
        'nsp_label': torch.tensor(nsp_label, dtype = torch.long),
    }
    
def main():
    project_root = Path(__file__).parent.parent
    raw_data_path = project_root / 'input.txt'
    processed_data_dir = project_root / 'data' / 'processed'
    tokenizer_path = project_root / 'data' / 'tokenizer.pkl'

    processed_data_dir.mkdir(parents = True, exist_ok = True)
    print('Step 1: Reading and Cleaning Text: ')
    with open(raw_data_path, 'r', encoding = 'utf-8') as f:
        raw_text = f.read()
    
    cleaned_text = clean_text(raw_text)

    try:
        nltk.data.find('tokenizers/punkt')
    except nltk.downloader.DownloadError:
        print("Download NLTK 'punkt' model...")
        nltk.download('punkt')

    sentences = nltk.sent_tokenize(cleaned_text)
    print(f"   ...found {len(sentences)} sentences.")

    print(f'Step 2: Building vocabulary (size = {VOCAB_SIZE}) and tokenizer...')
    tokenizer = Tokenizer.build(sentences, VOCAB_SIZE)
    with open(tokenizer_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f'   ...vocabulary and tokenizer saved to {tokenizer_path}')

    print("Step 3: Creating sentence pairs for NSP...")
    sentence_pairs = create_sentence_pairs(sentences)
    print(f'    ...created {len(sentence_pairs)} pairs.')

    print('Step 4: Generating and saving training examples...')
    training_examples = []
    for pair in tqdm(sentence_pairs, desc = 'Creating examples'):
        example = create_training_example(pair, tokenizer, MAX_SEQ_LEN, MASK_PROB)
        training_examples.append(example)
    
    output_file = processed_data_dir / 'bert_data.pkl'
    with open(output_file, 'wb') as f:
        pickle.dump(training_examples, f)

if __name__ == '__main__':
    main()