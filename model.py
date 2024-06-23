import torch
from torch import nn

device = 'cuda'

class CNN(nn.Module):
  def __init__(self, input_channels, hidden_size, max_seq_len=80):
    super(CNN, self).__init__()
    self.cnn = nn.Sequential(
        nn.Conv2d(input_channels, 64, kernel_size=(3,3), stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(kernel_size=(2,2)),
        nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(256),
        nn.ReLU(inplace=False),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(256, hidden_size, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(hidden_size),
        nn.ReLU(inplace=False),
    )
    self.project = nn.Linear(8*25, max_seq_len)

  def forward(self, x):
    
    x = self.cnn(x)
    # print("@@", x.shape)
    b, c, w, h = x.shape
    x = x.reshape(b, c, w*h)
    # print(x.shape)
    # x = self.project(x).permute(2, 0, 1) --> if CTC
    x = self.project(x).permute(0, 2, 1)

    return x

class BiLSTM(nn.Module):
    def __init__(self, input_size=512, hidden_size=64, 
                  num_classes=250):
        super(BiLSTM, self).__init__()

        self.lstm = nn.LSTM(input_size, hidden_size, 2,
                    bidirectional=True, batch_first=True) # batch_first ??
        self.fc = nn.Linear(hidden_size * 2, num_classes)
        self.act = nn.ReLU()
      
    def forward(self, x):
      x = self.lstm(x)[0] # (out, (hn, cn))
      # print("model --> rnn outptut ", x.shape)
      x = self.fc(x)
      return x

class NewNet(nn.Module):
    def __init__(self, input_channels, hidden_size, num_classes, max_seq_len):
      super(NewNet, self).__init__()
      self.cnn = CNN(input_channels, hidden_size, max_seq_len=max_seq_len)
      self.rnn = BiLSTM(input_size=hidden_size, hidden_size=hidden_size, 
                  num_classes=num_classes)

    def forward(self, x):
      # print("##", x.shape)
      x = self.cnn(x)
      # print(x.shape)
      x = self.rnn(x)
      # print("@@", x.shape)
      return x
        

# model = NewNet(1, 512, 76, 19).to(device)
# y = model(torch.zeros(32, 1, 64, 200).to(device))
# y.shape