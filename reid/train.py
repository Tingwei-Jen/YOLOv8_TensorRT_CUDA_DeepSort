import os
import argparse
import torch
import torchvision
from torch.utils.tensorboard import SummaryWriter

from read_category_dataset import read_and_split_data, CustomDataset
from model import ResNet18Reid

# Define the train function
def train(model, dataloader, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    running_loss = 0.0
    correct = 0
    for batch, (X, y) in enumerate(dataloader):
        # Upload data to device
        X, y = X.to(device), y.to(device)
        
        # Clear previous gradients
        optimizer.zero_grad()

        # Forward pass
        pred = model(X)

        # Calculate loss
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()

        # Update parameters
        optimizer.step()

        running_loss += loss.item()
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
        
        # Print the loss every 100 batches
        if batch % 100 == 0:
            loss, current = loss.item(), (batch + 1) * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

    running_loss /= num_batches
    correct /= size

    return correct, running_loss

# Define the test function
def test(model, dataloader, loss_fn, device):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for X, y in dataloader:
            # Upload data to device
            X, y = X.to(device), y.to(device)

            # Forward pass
            pred = model(X)

            # Calculate loss
            test_loss += loss_fn(pred, y).item()

            # Calculate accuracy
            correct += (pred.argmax(1) == y).type(torch.float).sum().item()

    test_loss /= num_batches
    correct /= size
    
    return correct, test_loss


# Define the save_checkpoint function
def save_checkpoint(model, optimizer, scheduler, epoch, folder_path):
    # Create the folder if it doesn't exist
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    
    # Define the filename
    filename = os.path.join(folder_path, f'checkpoint_epoch_{epoch}.pth')
    
    # Create the checkpoint dictionary
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
    }
    
    # Save the checkpoint
    torch.save(checkpoint, filename)
    
    # Print the message
    print(f"Checkpoint saved at epoch {epoch} to {filename}")


# Define the load_checkpoint function
def load_checkpoint(model, optimizer, scheduler, checkpoint_path, device):
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    start_epoch = checkpoint['epoch']+1 if 'epoch' in checkpoint else 1

    # Ensure all optimizer state tensors are on the correct device
    for state in optimizer.state.values():
        if isinstance(state, torch.Tensor):
            state.data = state.data.to(device)
        elif isinstance(state, dict):
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

    return model, optimizer, scheduler, start_epoch

def main(args):

    # tensorboard
    writer = SummaryWriter(log_dir='runs')

    # read data
    train_info, val_info, num_classes = read_and_split_data(args.data_dir, valid_rate=0.2)
    train_images_path, train_labels = train_info
    val_images_path, val_labels = val_info
    print("{} images for training.".format(len(train_images_path)))
    print("{} images for validation.".format(len(val_images_path)))

    # transform
    transform_train = torchvision.transforms.Compose([
        torchvision.transforms.RandomCrop((128, 64), padding=4),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    transform_val = torchvision.transforms.Compose([
        torchvision.transforms.Resize((128, 64)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # load dataset
    train_dataset = CustomDataset(
        image_paths=train_images_path,
        labels=train_labels,
        transform=transform_train
    )
    val_dataset = CustomDataset(
        image_paths=val_images_path,
        labels=val_labels,
        transform=transform_val
    )

    # dataloader
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
    )

    # model
    model = ResNet18Reid(num_classes=num_classes, reid=False)
    
    # visualize model on tensorboard
    dummy_input = torch.zeros(1, 3, 128, 64) 
    writer.add_graph(model, dummy_input)

    # loss, optimizer, scheduler
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # device setting
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # load pretrained weights
    start_epoch = 1
    if args.checkpoint_path:
        model, optimizer, scheduler, start_epoch = load_checkpoint(model, optimizer, scheduler, args.checkpoint_path, device)

    # model to device
    model.to(device) 

    # train
    for epoch in range(start_epoch, start_epoch + args.epochs):
        print(f"Epoch {epoch}\n-------------------------------")
        
        # train
        train_acc, train_loss = train(model, train_loader, loss_fn, optimizer, device)
        # print accuracy and loss
        print(f"Train Error: Accuracy: {(100*train_acc):>0.1f}%, Avg loss: {train_loss:>8f}")
        
        # test
        test_acc, test_loss = test(model, val_loader, loss_fn, device)
        # print accuracy and loss
        print(f"Test Error: Accuracy: {(100*test_acc):>0.1f}%, Avg loss: {test_loss:>8f}")
        
        # update learning rate
        scheduler.step()
        current_lr = scheduler.get_last_lr()[0]  # current learning rate
        # print current learning rate .6f
        print(f"Current learning rate: {current_lr:.6f}")

        # save checkpoint
        if epoch % 2 == 0:
            save_checkpoint(model, optimizer, scheduler, epoch, args.save_checkpoints_folder)

        # tensorboard
        writer.add_scalars('Loss', {'train': train_loss,'test': test_loss}, epoch)
        writer.add_scalars('Accuracy', {'train': train_acc,'test': test_acc}, epoch)
        writer.add_scalar('Learning Rate', current_lr, epoch)

    writer.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Train Resnet on Market1501")
    parser.add_argument("--data_dir", default='data', type=str)
    parser.add_argument('--epochs', type=int, default=40, help='number of epochs, default: 40')
    parser.add_argument('--batch_size', type=int, default=64, help='number of batch size, default: 64')
    parser.add_argument("--lr", default=0.01, type=float)
    parser.add_argument("--freeze_layer", default=0, type=int, help='freeze layer index, default: 0')
    parser.add_argument('--checkpoint_path', type=str, default=None, help='check_point path, default: None')
    parser.add_argument('--save_checkpoints_folder', type=str, default='checkpoints', help='checkpoints folder path')
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    main(args)
