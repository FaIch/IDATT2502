import torch
import numpy as np
import torch.nn as nn

# Emoji dictionary
emojis = {
    'hat': '\U0001F3A9',
    'rat': '\U0001F400',
    'cat': '\U0001F408',
    'flat': '\U0001F3E2',
    'matt': '\U0001F468',
    'cap': '\U0001F9E2',
    'son': '\U0001F466'
}


index_to_emoji = [value for _, value in emojis.items()]
index_to_char = [' ', 'h', 'a', 't', 'r', 'c', 'f', 'l', 'm', 'p', 's', 'o', 'n']

char_encodings = np.eye(len(index_to_char))
encoding_size = len(char_encodings)
emojies = np.eye(len(emojis))
emoji_encoding_size = len(emojies)

char_to_index = {char: idx for idx, char in enumerate(index_to_char)}
emoji_to_index = {emoji: idx for idx, emoji in enumerate(index_to_emoji)}

words = ['hat', 'rat', 'cat', 'flat', 'matt', 'cap', 'son']
emoji_labels = ['\U0001F3A9', '\U0001F400', '\U0001F408', '\U0001F3E2', '\U0001F468', '\U0001F9E2', '\U0001F466']

x_train = torch.zeros(len(words), 4, encoding_size, dtype=torch.float)
y_train = torch.zeros(len(words), 4, emoji_encoding_size, dtype=torch.float)

for i, word in enumerate(words):
    for j, char in enumerate(word.ljust(4, ' ')):
        x_train[i][j] = torch.tensor(char_encodings[char_to_index[char]], dtype=torch.float)

    emoji = emoji_labels[i]
    for j in range(4):
        y_train[i][j] = torch.tensor(emojies[emoji_to_index[emoji]], dtype=torch.float)


class EmojiLSTM(nn.Module):
    def __init__(self, encoding_size, emoji_encoding_size):
        super(EmojiLSTM, self).__init__()
        self.lstm = nn.LSTM(encoding_size, 128)  # 128 is the state size
        self.dense = nn.Linear(128, emoji_encoding_size)  # 128 is the state size

    def reset(self):
        zero_state = torch.zeros(1, 128)  # Changed to 2D tensor
        self.hidden_state = zero_state
        self.cell_state = zero_state

    def logits(self, x):
        out, (hn, cn) = self.lstm(x.view(x.size()[0], 1, -1),
                                  (self.hidden_state.view(1, 1, -1), self.cell_state.view(1, 1, -1)))
        self.hidden_state, self.cell_state = hn.view(1, -1), cn.view(1, -1)
        return self.dense(out.view(-1, 128))

    def f(self, x):
        return torch.softmax(self.logits(x), dim=1)

    def loss(self, x, y):
        return nn.functional.cross_entropy(self.logits(x), y.argmax(1))


model = EmojiLSTM(encoding_size, emoji_encoding_size)
optimizer = torch.optim.RMSprop(model.parameters(), 0.001)

for epoch in range(500):
    for i in range(x_train.size()[0]):
        model.reset()
        model.loss(x_train[i], y_train[i]).backward()
        optimizer.step()
        optimizer.zero_grad()


def generate_emoji(string):
    model.reset()
    for char in string.ljust(4, ' '):
        char_idx = char_to_index[char]
        y = model.f(torch.tensor(np.array([[char_encodings[char_idx]]]), dtype=torch.float))
    print(index_to_emoji[y.argmax(1)])


generate_emoji('rt')
generate_emoji('rats')
generate_emoji("h")
