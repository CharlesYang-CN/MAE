import torch.utils
from torch.utils.data import Dataset, DataLoader
import json
from PIL import Image
import torch
import torch.utils.data
from MAE import MAE_ViT
import torch
from torch.utils.data import random_split
from torchvision import transforms
import tqdm
import matplotlib.pyplot as plt

def split_train_test(dataset, train_ratio=0.8, seed=42):
    """
    将数据集划分为训练集和测试集，确保结果可复现
    
    参数:
    dataset (Dataset): 需要划分的PyTorch数据集
    train_ratio (float): 训练集占比（默认0.8）
    seed (int): 随机种子（默认42）
    
    返回:
    (train_dataset, test_dataset): 划分后的训练集和测试集
    """
    # 确保每次划分结果一致
    generator = torch.Generator().manual_seed(seed)
    
    # 计算划分数量
    train_size = int(train_ratio * len(dataset))
    test_size = len(dataset) - train_size
    
    # 进行随机划分
    train_dataset, test_dataset = random_split(
        dataset,
        [train_size, test_size],
        generator=generator
    )
    
    return train_dataset, test_dataset

class CustomImageDataset(Dataset):
    def __init__(self,json_path,transform = None):
        super().__init__()

        with open(json_path,'r',encoding='utf-8') as f:
            self.datas = json.load(f)

        
        self.transform = transform
        self.all_image_paths = []

        for data in self.datas:
            img_paths = data['localpaths_png']
            for img_path in img_paths:
                self.all_image_paths.append(img_path)
    def __len__(self):

        return len(self.all_image_paths)

    def __getitem__(self, index):
        try:
            img_path = self.all_image_paths[index]
            image = Image.open(img_path).convert('RGB')  # 使用 Image.open
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")  # 打印错误信息
            image = torch.zeros(3, 224, 224)
        return image

def train(train_dataloader, model, epoch, lr_scheduler, optim, device, save_path='model_best.pth'):
    # 初始化最佳损失为无穷大
    best_loss = float('inf')
    
    for e in range(epoch):
        model.train()
        losses = []
        
        # 训练循环
        for img in tqdm(iter(train_dataloader)):
            img = img.to(device)
            
            # 前向传播
            predicted_img, mask = model(img)
            
            # 计算损失
            loss = torch.mean((predicted_img - img) ** 2 * mask) / 0.75
            
            # 反向传播和优化
            loss.backward()
            optim.step()
            optim.zero_grad()
            
            # 记录损失
            losses.append(loss.item())
        
        # 更新学习率
        lr_scheduler.step()
        
        # 计算平均损失
        avg_loss = sum(losses) / len(losses)
        
        # 打印和记录日志
        with open('log.txt', 'a', encoding='utf-8') as f:
            f.write(f'In epoch {e}, average training loss is {avg_loss}.\n')
        print(f'In epoch {e}, average training loss is {avg_loss}.')
        
        # 保存最佳模型
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), save_path)
            print(f"New best model saved at epoch {e} with loss {best_loss}.")
            with open('log.txt', 'a', encoding='utf-8') as f:
                f.write(f"New best model saved at epoch {e} with loss {best_loss}.\n")
    
    return None

def test(test_dataloader, model, device, save_dir='reconstruction'):
    model.eval()
    losses = []
    with torch.no_grad():
        for i, img in enumerate(tqdm(iter(test_dataloader))):
            img = img.to(device)
            predicted_img, mask = model(img)
            loss = torch.mean((predicted_img - img) ** 2 * mask) / 0.75
            losses.append(loss.item())

            # 可视化第一批的前几张图像
            if i == 0:
                for j in range(min(3, img.size(0))):  # 保存前3张图像
                    orig_img = img[j].cpu().permute(1, 2, 0).numpy()
                    recon_img = predicted_img[j].cpu().permute(1, 2, 0).numpy()
                    plt.figure(figsize=(10, 5))
                    plt.subplot(1, 2, 1)
                    plt.imshow(orig_img)
                    plt.title('Original')
                    plt.subplot(1, 2, 2)
                    plt.imshow(recon_img)
                    plt.title('Reconstructed')
                    plt.savefig(f'{save_dir}/sample_{j}.png')
                    plt.close()

        avg_loss = sum(losses) / len(losses)
        print(f'Average test loss: {avg_loss}')
    return avg_loss
    
    
if __name__ == '__main__':
    json_path = ''
    epoch = 8
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像大小为 224x224
    transforms.ToTensor(),          # 转换为张量
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
    ])


    model = MAE_ViT(patch_size=16, emb_dim=384)
    model = model.to(device)

    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epoch)
    dataset = CustomImageDataset(json_path=json_path,transform=transform)
    train_dataset , test_dataset = split_train_test(dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    train(train_dataloader, model, epoch, lr_scheduler, optim)
    test(test_dataloader, model, device)
    
