import torch
import torch.nn as nn
from vector_quantize_pytorch import ResidualVQ

class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

class ResidualUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(in_channels, out_channels),
            nn.ELU(),
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, x):
        return x + self.layers(x)


class EncoderBlock(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            ResidualUnit(in_channels=input_channels,
                         out_channels=input_channels),
            nn.ELU(),
            ResidualUnit(in_channels=input_channels,
                         out_channels=input_channels),
            nn.ELU(),
            ResidualUnit(in_channels=input_channels,
                         out_channels=input_channels),
            nn.ELU(),
            nn.Linear(input_channels, out_channels)
        )

    def forward(self, x):
        return self.layers(x)


class DecoderBlock(nn.Module):
    def __init__(self, input_channels, out_channels):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(input_channels,out_channels),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),
            nn.ELU(),
            ResidualUnit(in_channels=out_channels, out_channels=out_channels),

        )

    def forward(self, x):
        return self.layers(x)


class Encoder(nn.Module):
    def __init__(self, C=768, D=128):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(C, C),
            nn.ELU(),
            EncoderBlock(C, out_channels=512),
            nn.ELU(),
            EncoderBlock(512, out_channels=256),
            nn.ELU(),
            EncoderBlock(256, out_channels=128),
            nn.ELU(),
            EncoderBlock(128, out_channels=D),
            nn.ELU(),
            nn.Linear(D, D)
        )
        # self.layers = nn.Sequential(
        #     nn.Linear(C, C),
        #     nn.ELU(),
        #     EncoderBlock(C, out_channels=512),
        #     nn.ELU(),
        #     EncoderBlock(512, out_channels=512),
        #     nn.ELU(),
        #     EncoderBlock(512, out_channels=D),
        #     nn.ELU(),
        #     EncoderBlock(D, out_channels=D),
        #     nn.ELU(),
        #     nn.Linear(D, D)
        # )
    def forward(self, x):
        return self.layers(x)


class Decoder(nn.Module):
    # def __init__(self, C, D):
    #     super().__init__()

    #     self.layers = nn.Sequential(
    #         nn.Linear(96, 96),
    #         nn.ELU(),
    #         DecoderBlock(input_channels=96, out_channels=128),
    #         nn.ELU(),
    #         DecoderBlock(128, out_channels=256),
    #         nn.ELU(),
    #         DecoderBlock(256, out_channels=512),
    #         nn.ELU(),
    #         DecoderBlock(512, out_channels=768),
    #         nn.ELU(),
    #         nn.Linear(768, 768)
    #     )
    def __init__(self, C, D):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Linear(D, 256),
            nn.ELU(),
            # DecoderBlock(input_channels=96, out_channels=128),
            # nn.ELU(),
            # DecoderBlock(96, out_channels=512),
            # nn.ELU(),
            DecoderBlock(256, out_channels=512),
            nn.ELU(),
            DecoderBlock(512, out_channels=768),
            nn.ELU(),
            nn.Linear(768, 768)
        )
    def forward(self, x):
        return self.layers(x)

class T5Stream(nn.Module):
    def __init__(self, C, D, num_quantizers, codebook_size):
        super().__init__()

        self.encoder = Encoder(C=C, D=D)
        self.quantizer = ResidualVQ(
            num_quantizers=num_quantizers, dim=D, codebook_size=codebook_size,
            kmeans_init=True, kmeans_iters=100, threshold_ema_dead_code=2, shared_codebook=True
        )
        self.decoder = Decoder(C=C, D=D)

    def forward(self, x):
        e = self.encoder(x)
        quantized, indices, commit_loss = self.quantizer(e)
        o = self.decoder(quantized)
        return o, indices, commit_loss


if __name__ == '__main__':
    input = torch.zeros(16, 768).cuda()
    enc_model = Encoder(768, 96).cuda()
    dec_model = Decoder(768, 96).cuda()

    enc_feat = enc_model(input)
    dec_feat = dec_model(enc_feat)
    print("?")
