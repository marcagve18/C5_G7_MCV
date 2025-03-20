from transformers import ResNetModel
from torch import nn
import torch
from torch.utils.data import DataLoader

from dataset import FoodDataset
from utils import get_train_val_test_annotations_split
from constants import CHAR2IDX, NUM_CHAR, TEXT_MAX_LEN


class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.encoder = ResNetModel.from_pretrained('microsoft/resnet-18').to(self.device)
        self.decoder = nn.GRU(512, 512, num_layers=1)
        self.proj = nn.Linear(512, NUM_CHAR)
        self.embed = nn.Embedding(NUM_CHAR, 512)

    def forward(self, img):
        batch_size = img.shape[0]
        feat = self.encoder(img)
        feat = feat.pooler_output.squeeze(-1).squeeze(-1).unsqueeze(0)  # 1, batch, 512
        start = torch.tensor(CHAR2IDX['<SOS>']).to(self.device)
        start_embed = self.embed(start)  # 512
        start_embeds = start_embed.repeat(batch_size, 1).unsqueeze(0)  # 1, batch, 512
        inp = start_embeds
        hidden = feat
        for t in range(TEXT_MAX_LEN - 1):  # rm <SOS>
            out, hidden = self.decoder(inp, hidden)
            inp = torch.cat((inp, out[-1:]), dim=0)  # N, batch, 512
    
        res = inp.permute(1, 0, 2)  # batch, seq, 512
        res = self.proj(res)  # batch, seq, 80
        res = res.permute(0, 2, 1)  # batch, 80, seq
        return res


if __name__ == "__main__":
    splits = get_train_val_test_annotations_split()
    train_annotations = splits["train"]  # Visualize training examples
    dataset = FoodDataset(train_annotations)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = Model()
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    
    with torch.no_grad():
        for batch in dataloader:
            images, _, _ = batch  # Assuming dataset returns images and labels
            images = images.to(device)
            print("Model Input Shape:", images.shape)
            outputs = model(images)
            print("Model Output Shape:", outputs.shape)  # Should be (batch, NUM_CHAR, TEXT_MAX_LEN)
            break  # Run for one batch to test inference
