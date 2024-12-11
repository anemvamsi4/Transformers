import csv
import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.utils import Config, BPETokenizer, load_dataset, LyricsDataset
from src.model import lyricGPT

# Training function
def train_model(model, train_dataloader, val_dataloader, config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = model.to(device)
    # model = torch.compile(model)
    # print("Using compiled model")
   
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    best_loss = float('inf')
    total_training_time = 0
    log_file = "./model-logs/training_log.csv"  
    log_fields = ["step", "time", "tokens_per_second", "train_loss", "val_loss"]

    # Initialize logging file
    with open(log_file, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=log_fields)
        writer.writeheader()

    step_count = 0  # Global step counter

    for epoch in range(config.max_epochs):
        print(f"\nStarting Epoch {epoch+1}/{config.max_epochs}")
        
        start_time = time.time()
        train_loss = 0.0
        total_tokens = 0

        model.train()  # Set model to training mode
        train_iterator = tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f"Epoch {epoch+1}")

        for batch_idx, (inputs, targets) in train_iterator:
            inputs = inputs.to(device)
            targets = targets.to(device)

            optimizer.zero_grad()
            logits, loss = model(inputs, targets)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            total_tokens += inputs.numel()
            step_count += 1

            train_iterator.set_postfix({
                'step': step_count,
                'loss': f"{loss.item():.4f}",
                'avg_loss': f"{train_loss/(batch_idx+1):.4f}"
            })

            # Perform saving, validation, and logging every 100 steps
            if step_count % 100 == 0:
                val_loss = perform_validation(model, val_dataloader, device)

                if val_loss < best_loss:
                    torch.save(model.state_dict(), './model-logs/best_model.pt')

                step_time = time.time() - start_time
                tokens_per_second = total_tokens / step_time 
                total_training_time += step_time

                # Log the metrics
                with open(log_file, 'a', newline='', encoding='utf-8') as csvfile:
                    writer = csv.DictWriter(csvfile, fieldnames=log_fields)
                    writer.writerow({
                        "step": step_count,
                        "time": step_time,
                        "tokens_per_second": tokens_per_second,
                        "train_loss": train_loss / (batch_idx + 1),
                        "val_loss": val_loss
                    })

        print(f"Epoch {epoch+1} Complete: Avg Train Loss: {train_loss/len(train_dataloader):.4f}")

    print("\nTraining Complete!")
    print(f"Total Training Time: {total_training_time/3600:.2f}hrs")

    return model


# Validation function
def perform_validation(model, val_dataloader, device) -> float:
    model.eval()  # Set model to evaluation mode
    val_loss = 0.0

    val_iterator = tqdm(val_dataloader, desc=f"Validation...", leave = False)
    with torch.no_grad():
        for inputs, targets in val_iterator:
            inputs = inputs.to(device)
            targets = targets.to(device)
            _, loss = model(inputs, targets)
            val_loss += loss.item()

    return val_loss / len(val_dataloader)

def trainer(config):

    data = load_dataset()

    # Initialize tokenizer
    tokenizer = BPETokenizer()
    tokenizer.load_vocab('./dataset/bpe_tokenizer')

    # Create dataset
    dataset = LyricsDataset(data, config.max_new_tokens)

    # Split Dataset into Train & Validation
    train_size = int(0.85 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])


    train_dataloader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=config.batch_size, shuffle=True)

    #training the model
    train_model(lyricGPT(config), train_dataloader, val_dataloader, config)

if __name__ == '__main__':
    config = Config()
    trainer(config)

