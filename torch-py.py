import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Dữ liệu mẫu, bảng chữ cái Hiragana
hiragana_alphabet = {
    'あ': 0, 'い': 1, 'う': 2, 'え': 3, 'お': 4,
    'か': 5, 'き': 6, 'く': 7, 'け': 8, 'こ': 9,
    'さ': 10, 'し': 11, 'す': 12, 'せ': 13, 'そ': 14,
    'た': 15, 'ち': 16, 'つ': 17, 'て': 18, 'と': 19,
    'な': 20, 'に': 21, 'ぬ': 22, 'ね': 23, 'の': 24,
    'は': 25, 'ひ': 26, 'ふ': 27, 'へ': 28, 'ほ': 29,
    'ま': 30, 'み': 31, 'む': 32, 'め': 33, 'も': 34,
    'や': 35, 'ゆ': 36, 'よ': 37,
    'ら': 38, 'り': 39, 'る': 40, 'れ': 41, 'ろ': 42,
    'わ': 43, 'を': 44,
    'ん': 45,
    'が': 46, 'ぎ': 47, 'ぐ': 48, 'げ': 49, 'ご': 50,
    'ざ': 51, 'じ': 52, 'ず': 53, 'ぜ': 54, 'ぞ': 55,
    'だ': 56, 'ぢ': 57, 'づ': 58, 'で': 59, 'ど': 60,
    'ば': 61, 'び': 62, 'ぶ': 63, 'べ': 64, 'ぼ': 65,
    'ぱ': 66, 'ぴ': 67, 'ぷ': 68, 'ぺ': 69, 'ぽ': 70,
    # Thêm các ký tự khác tương tự
}

# Xây dựng tập dữ liệu đào tạo mẫu
data_X = []
data_y = []

for char, label in hiragana_alphabet.items():
    # Tạo một array one-hot cho mỗi ký tự
    encoded = np.zeros(len(hiragana_alphabet))
    encoded[label] = 1
    data_X.append(encoded)
    data_y.append(label)

data_X = np.array(data_X, dtype=np.float32)
data_y = np.array(data_y, dtype=np.long)

# Xây dựng mô hình neural network sử dụng PyTorch
class HiraganaModel(nn.Module):
    def __init__(self):
        super(HiraganaModel, self).__init__()
        self.fc1 = nn.Linear(len(hiragana_alphabet), 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, len(hiragana_alphabet))

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = HiraganaModel()

# Định nghĩa hàm mất mát và bộ tối ưu
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Đào tạo mô hình
num_epochs = 100
for epoch in range(num_epochs):
    inputs = torch.tensor(data_X)
    labels = torch.tensor(data_y, dtype=torch.long)

    optimizer.zero_grad()
    outputs = model(inputs)
    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    if (epoch+1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# Lưu mô hình
torch.save(model.state_dict(), 'pytorch_hiragana_model.pth')
