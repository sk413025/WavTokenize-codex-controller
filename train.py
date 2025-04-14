import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

from pytorch_lightning.cli import LightningCLI, ArgsType
from utils.training_logger import TrainingLogger


def cli_main(args: ArgsType = None):
    # breakpoint()
    cli = LightningCLI(args=args)
    # breakpoint()
    cli.trainer.fit(model=cli.model, datamodule=cli.datamodule)


def train(model, train_loader, optimizer, criterion, num_epochs, device):
    logger = TrainingLogger()
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            # ...existing code...
            total_loss += loss.item()
        
        # Calculate average loss for the epoch
        avg_loss = total_loss / len(train_loader)
        logger.log_epoch(epoch, avg_loss)
        
        print(f'Epoch {epoch}: Average loss = {avg_loss:.4f}')
        
        # 每个epoch结束后保存loss图
        logger.plot_losses(save_path=f'logs/loss_plot_epoch_{epoch}.png')
    
    # 训练结束后保存最终的loss图
    logger.plot_losses(save_path='logs/final_loss_plot.png')


if __name__ == "__main__":
    cli_main()
