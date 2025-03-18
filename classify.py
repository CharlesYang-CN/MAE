import torch
from einops.layers.torch import Rearrange
from einops import repeat, rearrange
import json
from PIL import Image
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from sklearn.preprocessing import LabelEncoder
import tqdm
import logging
logging.basicConfig(filename='log.txt', level=logging.INFO, filemode='a', encoding='utf-8')


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
    def __init__(self, json_file,transform=None):
        """
        Args:
            json_file (string): JSON文件路径，包含了文件路径和标签。
            root_dir (string): 图片文件的根目录。
            transform (callable, optional): 图像预处理和数据增强操作。
        """
        with open(json_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)

        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        for item in self.data:
            file_paths = item['localpaths_png']
            label = item['label']
            
            # 过滤出包含 'PAA.png' 的文件路径
            AA_file_path = [path for path in file_paths if path.endswith('PAA.png')]
            #AB_files = [path for path in file_paths if path.endswith('PAB.png')]
            #DetailFiles = [path for path in file_paths if not path.endswith('PAA.png') and not path.endswith('PAA.png')]

            # 将图片路径和标签加入到列表
            self.image_paths.extend(AA_file_path)
            self.labels.extend([label] * len(AA_file_path))

        self.label_encoder = LabelEncoder()
        self.labels = self.label_encoder.fit_transform(self.labels)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        img_name = self.image_paths[idx]
        label = self.labels[idx]

        try:
            # 使用PIL打开图像
            image = Image.open(img_name).convert('RGB')
            # 进行变换（例如归一化等）
            if self.transform:
                image = self.transform(image)
        except Exception as e:
            print(f"Error loading {img_name}: {e}")  # 打印错误信息
            image = torch.zeros(3, 224, 224)
        
        return image, label

class ViT_Classifier(torch.nn.Module):
    def __init__(self,encoder, num_classes=5) -> None:
        super().__init__()
        self.cls_token = encoder.cls_token
        self.pos_embedding = encoder.pos_embedding
        self.patchify = encoder.patchify
        self.transformer = encoder.transformer
        self.layer_norm = encoder.layer_norm
        self.head = torch.nn.Linear(self.pos_embedding.shape[-1], num_classes)

        # 冻结除 head 外的所有参数
        for name, param in self.named_parameters():
            if 'head' not in name:  # 仅保留 head 层的参数可训练
                param.requires_grad = False

    def forward(self, img):
        patches = self.patchify(img)
        patches = rearrange(patches, 'b c h w -> (h w) b c')
        patches = patches + self.pos_embedding
        patches = torch.cat([self.cls_token.expand(-1, patches.shape[1], -1), patches], dim=0)
        patches = rearrange(patches, 't b c -> b t c')
        features = self.layer_norm(self.transformer(patches))
        features = rearrange(features, 'b t c -> t b c')
        logits = self.head(features[0])
        return logits
    
def validate(model, dataloader, device):
    """计算验证集准确率"""
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for imgs, labels in dataloader:
            imgs, labels = imgs.to(device), labels.to(device)
            outputs = model(imgs)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = correct / total
    model.train()
    return accuracy

def train(train_dataloader, val_dataloader, model, epoch, lr_scheduler, optim, device, save_path='model_best.pth'):
    best_acc = 0.0
    count = 0  # 累计处理的图片数量

    for e in range(epoch):
        model.train()
        losses = []

        for img, label in tqdm(train_dataloader, desc=f'Epoch {e}'):
            img, label = img.to(device), label.to(device)
            logits = model(img)
            loss = F.cross_entropy(logits, label)
            
            optim.zero_grad()
            loss.backward()
            optim.step()
            
            losses.append(loss.item())
            
            # 更新计数器并触发验证
            count += img.size(0)
            while count >= 2000:
                current_acc = validate(model, val_dataloader, device)
                if current_acc > best_acc:
                    best_acc = current_acc
                    torch.save(model.state_dict(), save_path)
                    print(f"\nNew best model saved at epoch {e} with accuracy {best_acc:.4f}")
                    logging.info(f"\nNew best model saved at epoch {e} with accuracy {best_acc:.4f}")
                count -= 2000  # 重置计数器，保留剩余图片数

        # 更新学习率并打印损失
        lr_scheduler.step()
        avg_loss = sum(losses) / len(losses)
        print(f'\nEpoch {e} average training loss: {avg_loss:.4f}')
        logging.info(f'\nEpoch {e} average training loss: {avg_loss:.4f}')

    
    return None

def test(test_dataloader, model, device):
    """
    Evaluate the model on the test dataset.
    
    Args:
        test_dataloader (DataLoader): DataLoader for the test dataset
        model (nn.Module): Trained model
        device (str): Device to run the evaluation on ('cuda' or 'cpu')
    
    Returns:
        float: Accuracy of the model on the test set
    """
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    with torch.no_grad():  # Disable gradient computation
        for img, label in tqdm(iter(test_dataloader)):
            img = img.to(device)
            label = label.to(device)
            logits = model(img)
            _, predicted = torch.max(logits, 1)  # Get predicted class
            total += label.size(0)
            correct += (predicted == label).sum().item()
    accuracy = correct / total
    print(f'Test Accuracy: {accuracy * 100:.2f}%')
    logging.info(f'Test Accuracy: {accuracy * 100:.2f}%')
    return accuracy


if __name__ == '__main__':
    json_file = ''
    batch_size = 1
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    mae_model = MAE_ViT().to(device)  # Assuming MAE_ViT is the original model class
    mae_model.load_state_dict(torch.load('model_best.pth'))
    encoder = mae_model.encoder
    model = ViT_Classifier(encoder, num_classes=5).to(device)
    transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = CustomImageDataset(json_file=json_file , transform=transform)
    train_dataset , test_dataset = split_train_test(dataset)
    train_dataloader = torch.utils.data.DataLoader(train_dataset,batch_size=batch_size,shuffle=True)
    test_dataloader = torch.utils.data.DataLoader(test_dataset,batch_size=batch_size,shuffle=False)

    epoch = 10
    optim = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.05)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=epoch)
    
    train(train_dataloader, model, epoch, lr_scheduler, optim)
    test(test_dataloader, model, device)
