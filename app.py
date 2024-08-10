import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import gradio as gr

# Define the Generator network (same as in your training script)
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(100, 512, 4, 1, 0, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )

    def forward(self, input):
        return self.main(input)

# Load the pre-trained model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
netG = Generator().to(device)
netG.load_state_dict(torch.load(r'generator.pth', map_location=device))
netG.eval()

# Function to generate anime faces
def generate_anime_face():
    noise = torch.randn(1, 100, 1, 1, device=device)
    with torch.no_grad():
        fake = netG(noise).detach().cpu()
    # Denormalize and convert to PIL image
    fake = (fake + 1) / 2
    fake_image = transforms.ToPILImage()(fake.squeeze(0))
    return fake_image

# Create Gradio interface
iface = gr.Interface(
    fn=generate_anime_face,
    inputs=[],
    outputs=gr.Image(type="pil"),
    title="Anime Face Generator",
    description="Generate anime faces using a trained GAN model."
)

if __name__ == "__main__":
    iface.launch()
    
